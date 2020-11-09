from dataclasses import dataclass
from typing import List, NamedTuple
import numpy as np
import math


@dataclass
class Datum:
    line_idx: int
    src: List[str]
    tgt: List[str]
    trace: float
    num_rows: int
    num_cols: int
    attention: np.ndarray


def parse_attention_file(filename) -> List[Datum]:
    with open(filename) as f:
        data = f.read().splitlines()
        item = sum(1 for line in data if line == "")
        if data and item == 0:
            item = 1
    data = parse_lines(data, item)
    return data


def parse_lines(dataset: List[str], num_items: int) -> List[Datum]:
    jump = 0
    data = []
    for i in range(num_items):
        line = i + jump
        _, tgt, trace, src, cxr = dataset[line].split(" ||| ")
        col, row = [int(x) for x in cxr.split()]
        start_of_tensor = line + 1
        end_of_tensor = start_of_tensor + row
        att_str = "; ".join(dataset[start_of_tensor:end_of_tensor])
        attention = np.array(np.mat(att_str))
        data.append(Datum(line, src.split(), tgt.split(), float(trace), row, col, attention))
        jump = jump + row + 1
    return data


def plot_attentions(
    tgt: List[str], src: List[str], att_matrix: np.ndarray, idx=None, figsize=(8, 6)
):
    import matplotlib.pyplot as plt
    assert all([a == b for a, b in zip(att_matrix.shape, (len(tgt), len(src)))])

    fig, ax = plt.subplots(figsize=figsize)

    cs = ax.imshow(att_matrix, cmap="rainbow")
    if idx is not None:
        ax.set_title("Attention: Item {}".format(idx))
    plt.xticks(np.arange(len(src)))
    plt.yticks(np.arange(len(tgt)))
    plt.tight_layout()
    ax.set_xticklabels(src, rotation="vertical", fontsize=10, va="top", ha="center")
    ax.set_yticklabels(tgt, rotation="horizontal", fontsize=10, va="center", ha="right")
    ax.tick_params(axis="both", direction="out", which="major", pad=5)
    plt.colorbar(cs)
    fig.tight_layout()
    return fig


if __name__ == "__main__":

    attention_path = "sample.attentions"  # attentions file path
    line_number = 0  # number of source sentence to visualise
    data = parse_attention_file(attention_path)
    d = data[line_number]
    fig = plot_attentions(d.tgt, d.src, d.attention)
    fig.savefig("attention.png")
