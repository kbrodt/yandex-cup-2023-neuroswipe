import numpy as np
import torch
import torch.utils.data as data


class Vocab:
    def __init__(self, voc_path=None, words=None):
        words = [] if words is None else words
        self.max_len = 0
        if voc_path is not None:
            with open(voc_path) as f:
                for l in f:
                    word = l.rstrip().split("\t")[0]
                    words.append(word.strip().lower())
                    if len(words[-1]) > self.max_len:
                        self.max_len = len(words[-1])

        self.pad_i = 0
        self.sos_i = 1
        self.eos_i = 2
        self.unk_i = 3
        self.idx2token = ["<PAD>", "<SOS>", "<EOS>", "<UNK>"]
        self.idx2token.extend(sorted(set("".join(words))))
        self.token2idx = {
            token: i
            for i, token in enumerate(self.idx2token)
        }
        self.words = words

    def __len__(self):
        return len(self.idx2token)

    def to_idx(self, word):
        return [self.token2idx[c] for c in word]

    def to_token(self, idxs):
        return "".join(self.idx2token[i] for i in idxs if i not in [self.pad_i, self.sos_i, self.eos_i, self.unk_i])

    def to_idxb(self, words, max_len=None):
        if max_len is None:
            max_len = max(map(len, words)) + 1

        #x = np.full((len(words), max_len), self.pad_i, dtype="int64")  # not stable training lstm!
        m = np.zeros((len(words), max_len), dtype="bool")
        x = np.full((len(words), max_len), self.eos_i, dtype="int64")
        seq_lens = []
        for i, word in enumerate(words):
            idx = self.to_idx(word)
            idx.append(self.eos_i)
            sl = len(idx)
            x[i, :sl] = idx
            m[i, sl:] = True
            seq_lens.append(sl)

        return x, seq_lens, m

    def to_tokenb(self, ids):
        words = []
        for i in ids:
            word = self.to_token(i)
            words.append(word)

        return words


class PairDataset(data.Dataset):
    def __init__(self, df, voc, is_train=False):
        self.df = df
        self.voc = voc
        self.alph = self.voc.idx2token[5:]
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        x = item.inp
        #if self.is_train:
        #    x = list(x)
        #    n = random.randint(0, int(0.3 * len(x)))
        #    sam = np.random.choice(len(x), size=n, replace=False)
        #    for ind in sam:
        #        x[ind] = random.choice(self.alph)

        #    x = "".join(x)

        y = item.out

        return x, y

    def collate_fn(self, x):
        x, y = list(zip(*x))

        x, lx, _  = self.voc.to_idxb(x)
        x = torch.from_numpy(x)
        lx = torch.tensor(lx, dtype=torch.int64)

        y, ly, _  = self.voc.to_idxb(y)
        y = torch.from_numpy(y)
        ly = torch.tensor(ly, dtype=torch.int64)

        return x, lx, y, ly


class CurveDataset(data.Dataset):
    def __init__(self, df, grids):
        self.df = df
        self.grids = grids

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        x = item.src_cx
        x = [float(c.strip(' []')) for c in x.split(',')]

        y = item.src_cy
        y = [float(c.strip(' []')) for c in y.split(',')]

        src_curve = np.array(list(zip(x, y)), dtype="float32")

        word = item.out
        grid_t = item.grid_t
        grid = self.grids[grid_t]
        tgt_curve = grid.get_centered_curve(word)

        src_curven = grid.normalize(src_curve)
        src_curven = np.clip(src_curven, -1, 1)
        tgt_curven = grid.normalize(tgt_curve)
        tgt_curven = np.clip(tgt_curven, -1, 1)

        src_curve = np.vstack(
            [
                src_curven,
                self.grids["eos"],
            ]
        ).astype("float32")
        tgt_curve = np.vstack(
            [
                tgt_curven,
                self.grids["eos"],
            ]
        ).astype("float32")

        return src_curve, tgt_curve, word, grid_t

    def collate_fn(self, x):
        x, y, w, g = list(zip(*x))

        lx = list(map(len, x))
        ly = list(map(len, y))

        x = np.concatenate(x, axis=0)
        y = np.concatenate(y, axis=0)

        x = torch.from_numpy(x)
        y = torch.from_numpy(y)

        return x, lx, y, ly, w, g


class CurveWordDataset(data.Dataset):
    def __init__(self, df, grids, voc, is_train=False):
        self.df = df
        self.grids = grids
        self.voc = voc
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        item = self.df.iloc[index]
        x = item.src_cx
        x = [float(c.strip(' []')) for c in x.split(',')]

        y = item.src_cy
        y = [float(c.strip(' []')) for c in y.split(',')]

        src_curve = np.array(list(zip(x, y)), dtype="float32")

        #if self.is_train:
        #    n = src_curve.shape[0]
        #    r = np.random.uniform(0.6, 1)
        #    idx = np.random.choice(n, size=int(r * n), replace=False)
        #    idx.sort()
        #    src_curve = src_curve[idx]

        grid_t = item.grid_t
        grid = self.grids[grid_t]

        src_curven = grid.normalize(src_curve)

        if self.is_train:
            src_curven += 0.02 * np.random.randn(*src_curven.shape)

        src_curven = np.clip(src_curven, -1, 1)

        src_curve = np.vstack(
            [
                src_curven,
                self.grids["eos"],
            ]
        ).astype("float32")

        y = item.out
        m = torch.zeros(len(src_curve), dtype=torch.bool)

        return src_curve, y, m

    def collate_fn(self, x):
        x, y, m = list(zip(*x))
        t = max(map(len, m))
        mask = torch.stack(
            [
                torch.nn.functional.pad(
                    o,
                    (0, t - o.shape[0]),
                    mode="constant",
                    value=True,
                )
                for o in m
            ],
            dim=0,
        ) # [b, t]

        lx = list(map(len, x))

        x = np.concatenate(x, axis=0)
        x = torch.from_numpy(x)

        y, ly, my = self.voc.to_idxb(y)
        y = torch.from_numpy(y)
        ly = torch.tensor(ly, dtype=torch.int64)
        my = torch.from_numpy(my)

        return x, lx, mask, y, my


if __name__ == "__main__":
    voc = Vocab(words=["asd", "qwe", ""])
    words = ["aqe", "e"]
    print(words)
    x, seq_lens, m = voc.to_idxb(words)
    print(x)
    print(seq_lens)
    print(m)
    print(voc.to_tokenb(x))
    x = torch.from_numpy(x)
    seq_lens = torch.tensor(seq_lens, dtype=torch.int64)
    seq_lens, perm_idx = seq_lens.sort(0, descending=True)
    x = x[perm_idx]
    emb = torch.nn.Embedding(
        len(voc),
        4,
        padding_idx=voc.pad_i
    )
    logits_pre = emb(x)

    logits = torch.nn.utils.rnn.pack_padded_sequence(logits_pre, seq_lens, batch_first=True)
    print(logits)

    logits_post, input_sizes = torch.nn.utils.rnn.pad_packed_sequence(logits, batch_first=True)
    assert torch.allclose(logits_pre, logits_post)
    assert torch.allclose(input_sizes, seq_lens)
