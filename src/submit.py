import os
import json

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import tqdm
import pandas as pd
from sklearn.model_selection import KFold

import models
import dataset
from grid import Grid


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-df",
        type=str,
        default="./data/train_c.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--grid-path",
        type=str,
        default="./data/grid.json",
        help="path to grid",
    )
    parser.add_argument(
        "--voc-path",
        type=str,
        default="./data/voc.txt",
        help="path to voc",
    )
    parser.add_argument("--checkpoint-dir", type=str, default="logs")

    parser.add_argument("--backbone", type=str, default="seq2seq")

    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--n-layers", type=int, default=1)
    parser.add_argument("--label-smoothing", type=int, default=0.05)

    parser.add_argument(
        "--num-workers", type=int, help="number of data loader workers", default=8,
    )
    parser.add_argument("--batch-size", type=int, help="batch size", default=32)
    parser.add_argument(
        "--random-state",
        type=int,
        help="random seed",
        default=314159,
    )

    parser.add_argument(
        "--n-folds",
        type=int,
        help="number of folds",
        default=10,
    )
    parser.add_argument(
        "--fold",
        type=int,
        help="fold",
        default=0,
    )

    parser.add_argument(
        "--load", type=str, default="", help="path to pretrained model weights"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="path to pretrained model to resume training",
    )
    parser.add_argument("--fp16", action="store_true", help="fp16 training")
    parser.add_argument("--ft", action="store_true", help="finetune")

    args = parser.parse_args(args=args)

    return args


@torch.inference_mode()
def epoch_step_val(loader, model, metric, max_len, beam_width=4):
    model.eval()

    pbar = tqdm.tqdm(total=len(loader), leave=False, mininterval=2)

    for inp, inp_len, src_padding_mask, target, tgt_padding_mask in loader:
        inp = inp.cuda(non_blocking=True)

        if beam_width > 1:
            logits, _ = models.beam_search(
                model,
                (inp, inp_len),
                src_padding_mask,
                predictions=max_len,
                beam_width=beam_width,
            )
            metric["mrr"].update_bs(logits, target)
        else:
            logits = model(
                (inp, inp_len),
                src_padding_mask=src_padding_mask,  # .cuda(non_blocking=True),
            )
            metric["mrr"].update(logits, target)

        torch.cuda.synchronize()

        pbar.update()

    pbar.close()


class Metric:
    def __init__(self, grids, voc):
        self.grids = grids
        self.voc = voc
        self.scores = []
        self.df = []
        self.clean()

    def clean(self):
        self.scores.clear()
        self.df.clear()

    def update(self, preds, targets):
        # preds   [b, t, c]
        # targets [b * t, 2]
        assert len(preds) == len(targets)

        target = targets.cpu().numpy()
        target = self.voc.to_tokenb(target)

        _, preds = preds.topk(1, dim=-1)
        preds = preds.squeeze(-1)
        preds = preds.cpu().numpy()

        preds = self.voc.to_tokenb(preds)
        for a, b in zip(preds, target):
            self.scores.append(a == b)
            self.df.append((b, a, a, a, a))

    def update_bs(self, preds, targets):
        # preds   [b, n, t]
        # targets [b, t]
        assert len(preds) == len(targets), (preds.shape, targets.shape)

        target = targets.cpu().numpy()
        target = self.voc.to_tokenb(target)

        preds = preds.cpu().numpy()

        ps = []
        for c in range(preds.shape[1]):
            p = self.voc.to_tokenb(preds[:, c])
            ps.append(p)

        for *a, b in zip(*ps, target):
            self.df.append((b, *a))
            self.scores.append(sum(
                w * (x == b)
                for x, w in zip(a[:4], [1, 0.1, 0.09, 0.08])
            ))

    def evaluate(self, path=None):
        if len(self.scores) == 0:
            return 0.0

        if path is not None:
            columns = ["t"] + [f"pred_{i}" for i in range(len(self.df[0]) - 1)]
            df = pd.DataFrame(
                self.df,
                columns=columns,
            )
            path = Path(path)
            df.to_csv(path, index=False)
            #df[columns[1:]].to_csv(
            #    path.with_name(f"{path.stem}_fin{path.suffix}"),
            #    index=False,
            #    header=False,
            #)

        score = sum(self.scores) / len(self.scores)

        return score


def train_dev_split(df, args):
    skf = KFold(
        n_splits=args.n_folds, shuffle=True, random_state=args.random_state
    )
    df["fold"] = None
    n_col = len(df.columns) - 1
    for fold, (_, dev_index) in enumerate(skf.split(df)):
        df.iloc[dev_index, n_col] = fold

    train, dev = (
        df[df.fold != args.fold].copy(),
        df[df.fold == args.fold].copy(),
    )
    _dev = dev.sample(n=10_000, random_state=args.random_state)
    #_dev = dev.sample(n=1000, random_state=args.random_state)
    #_dev = dev[~dev.index.isin(_dev.index)].sample(n=10_000, random_state=args.random_state)

    train = pd.concat(
        [
            train,
            #dev[~dev.index.isin(_dev.index)]
        ]
    )
    dev = _dev
    assert len(set(train.index) & set(dev.index)) == 0

    return train, dev


def read_df(path):
    df = pd.read_csv(path)#, nrows=3_000_000)
    if "train" in path:
        df = df.dropna()
        df = df.drop_duplicates(subset=["inp", "out"])

    return df


def train(args):
    torch.backends.cudnn.benchmark = True

    _df = read_df(args.train_df)
    is_train = "train" in args.train_df
    if "test" in args.train_df:
        _df.out = "кек"
    df = pd.concat(
        [
            _df,
            #_df_sa,
        ],
    )

    with open(args.grid_path) as fin:
        grids = [json.loads(l) for l in fin]

    grids = {
        grid["grid_name"]: Grid(grid)
        for grid in grids
    }
    grids["pad"] = 0.0
    grids["eos"] = [[1, 1.]]

    voc = dataset.Vocab(args.voc_path)
    pd.DataFrame(voc.words, columns=["words"]).to_csv("vocab_words.csv", index=False)

    path_to_resume = Path(args.load).expanduser()
    print(f"=> loading resume checkpoint '{path_to_resume}'")
    checkpoint = torch.load(
        path_to_resume,
        map_location="cpu",
    )

    args = checkpoint["args"]
    print(args.n_tokens)
    print(args.max_len)
    assert args.max_len == voc.max_len + 1

    model = build_model(args, voc, grids)
    model = model.cuda()

    nn.modules.utils.consume_prefix_in_state_dict_if_present(checkpoint["state_dict"], "module.")
    model.load_state_dict(checkpoint["state_dict"])
    print(
        f"=> resume from checkpoint '{path_to_resume}' (epoch {checkpoint['epoch']})"
    )

    print(args)
    if is_train:
        _, val_df = train_dev_split(_df, args)
    else:
        val_df = _df

    print(val_df)

    val_dataset = dataset.CurveWordDataset(
        val_df,
        grids,
        voc,
        is_train=False,
    )

    val_sampler = None

    args.num_workers = min(args.batch_size, 8)
    val_batch_size = args.batch_size
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        collate_fn=val_dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=False,
        persistent_workers=True,
        drop_last=False,
    )

    metric = {
        "mrr": Metric(grids, voc),
    }

    for m in metric.values():
        m.clean()

    epoch_step_val(
        val_loader,
        model,
        metric,
        max_len=args.max_len,
        beam_width=4,
    )

    dev_scores = {}
    for key, m in metric.items():
        dev_scores[key] = m.evaluate("submission.csv")

    print(dev_scores)
    torch.cuda.empty_cache()


def build_model(args, voc, grids):
    if "lstm" in args.backbone or "gru" in args.backbone:
        emb_enc = models.Board(
            m=args.m,
            n=args.n,
            emb_size=args.emb_dim,
            pad_i=grids["pad"],
        )
        emb_dec = nn.Embedding(
            args.n_tokens,
            args.emb_dim,
            padding_idx=voc.pad_i,
        )
        model = models.Seq2Seq(
            encoder=models.EncoderRNN(
                input_size=args.n_tokens,
                emb_size=args.emb_dim,
                hidden_size=args.hidden_size,
                num_layers=args.n_layers,
                emb=emb_enc,
                dropout_p=0.1,
            ),
            decoder=models.AttnDecoderRNN(
                hidden_size=args.hidden_size,
                emb_size=args.emb_dim,
                output_size=args.n_tokens,
                max_len=args.max_len,
                num_layers=args.n_layers,
                sos_idx=voc.sos_i,
                emb=emb_dec,
                dropout_p=0.1,
            )
        )
    elif "transformer" in args.backbone:
        model = models.Seq2SeqTransformer(
            emb_size=args.emb_dim,
            emb_src=models.Board(
                m=args.m,
                n=args.n,
                emb_size=args.emb_dim,
                pad_i=grids["pad"],
            ),
            emb_tgt=nn.Embedding(
                args.n_tokens,
                args.emb_dim,
                padding_idx=voc.pad_i,
            ),
            n_tokens=args.n_tokens,
            max_len=args.max_len,
            sos_idx=voc.sos_i,
        )

    return model


def main():
    args = parse_args()

    train(args)


if __name__ == "__main__":
    main()
