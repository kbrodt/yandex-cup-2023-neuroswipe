import argparse
import functools
import itertools
import json
import multiprocessing
from pathlib import Path

import pandas as pd
import numpy as np
import tqdm

from grid import Grid


def parse_args(args=None):
    p = argparse.ArgumentParser()
    p.add_argument(
        '--file-path',
        type=str,
        default="./data/train.jsonl",
        required=True,
    )

    p.add_argument(
        '--output-path',
        type=str,
        required=True,
    )

    p.add_argument(
        '--num-workers',
        type=int,
        default=1,
    )

    args = p.parse_args(args)

    return args


def process_sample(line, valid=None):
    i, line = line
    sample = json.loads(line)

    grid_info = sample["curve"]["grid"]
    grid = Grid(grid_info)
    curve_data = sample["curve"]
    x = curve_data["x"]
    y = curve_data["y"]

    curve = np.array(list(zip(x, y)))
    src_word = "".join(
        k
        for k, _ in itertools.groupby(
            "".join(
                grid.get_the_nearest_hitbox(x, y)
                for x, y in curve
            )
        )
    )

    if valid is not None:
        tgt_word = valid[i]
    else:
        tgt_word = sample.get("word", "")

    return src_word, tgt_word, x, y, grid_info["grid_name"]


def main():
    args = parse_args()

    valid = None
    if "valid" in args.file_path:
        with open(Path(args.file_path).with_suffix(".ref")) as fin:
            valid = [line.strip() for line in fin]

    with open(args.file_path, "r") as file:
        if args.num_workers > 1:
            with multiprocessing.Pool(args.num_workers) as p:
                output = list(
                    p.imap(
                        func=functools.partial(process_sample, valid=valid),
                        iterable=tqdm.tqdm(enumerate(file))
                    )
                )
                src_words, tgt_words, src_cx, src_cy, grid_t = zip(*output)
        else:
            src_words = []
            tgt_words = []
            src_cx = []
            src_cy = []
            grid_t = []
            for i, line in enumerate(tqdm.tqdm(file)):
                src_word, tgt_word, x, y, grid_name = process_sample((i, line), valid=valid)
                src_words.append(src_word)
                tgt_words.append(tgt_word)
                src_cx.append(x)
                src_cy.append(y)
                grid_t.append(grid_name)

    pd.DataFrame({
        "inp": src_words,
        "out": tgt_words,
        "src_cx": src_cx,
        "src_cy": src_cy,
        "grid_t": grid_t,
    }).to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()
