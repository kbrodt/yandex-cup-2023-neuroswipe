import argparse

import pandas as pd
from rapidfuzz.process import cdist
from rapidfuzz.distance import (
    DamerauLevenshtein,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sub-path",
        type=str,
        default="./data/submission.csv",
        help="path to train df",
    )
    parser.add_argument(
        "--voc-path",
        type=str,
        default="./data/vocab_words.csv",
        help="path to vocab df",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default="./data/out.csv",
        help="path to output file",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=20,
    )
    parser.add_argument("--use-beam", action="store_true", help="use beam search")

    args = parser.parse_args(args=args)

    return args


def single(args):
    df = pd.read_csv(args.sub_path)
    vocab = pd.read_csv(args.voc_path)

    m = ~df.pred_0.isin(vocab.words)

    cd = cdist(
        df.loc[m, "pred_0"],
        vocab.words,
        scorer=DamerauLevenshtein.normalized_distance,
        workers=args.n_workers,
    )
    df.loc[m, "pred_0"] = vocab.words[cd.argmin(1)].values

    df.to_csv(args.out_path, index=False)


def beam(args):
    df = pd.read_csv(args.sub_path)
    vocab = pd.read_csv(args.voc_path)

    m = ~df.pred_2.isin(vocab.words)
    df.loc[m, "pred_2"] = df.loc[m, "pred_3"]

    m = ~df.pred_1.isin(vocab.words)
    df.loc[m, "pred_1"] = df.loc[m, "pred_2"]
    m = ~df.pred_1.isin(vocab.words)
    df.loc[m, "pred_1"] = df.loc[m, "pred_3"]

    m = ~df.pred_0.isin(vocab.words)
    df.loc[m, "pred_0"] = df.loc[m, "pred_1"]
    m = ~df.pred_0.isin(vocab.words)
    df.loc[m, "pred_0"] = df.loc[m, "pred_2"]
    m = ~df.pred_0.isin(vocab.words)
    df.loc[m, "pred_0"] = df.loc[m, "pred_3"]

    for i in range(1):
        m = ~df[f"pred_{i}"].isin(vocab.words)
        cd = cdist(
            df.loc[m, f"pred_{i}"],
            vocab.words,
            scorer=DamerauLevenshtein.normalized_distance,
            workers=args.n_workers,
        )
        df.loc[m, f"pred_{i}"] = vocab.words[cd.argmin(1)].values

    df.to_csv(args.out_path, index=False)


def main():
    args = parse_args()
    if args.use_beam:
        beam(args)
    else:
        single(args)


if __name__ == "__main__":
    main()
