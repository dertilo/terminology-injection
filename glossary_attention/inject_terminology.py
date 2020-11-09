import argparse
from collections import defaultdict
from dataclasses import dataclass

import numpy
import re

import numpy as np
from sklearn.preprocessing import normalize
from typing import List, Dict, Tuple

from glossary_attention.read_plot_attentions import (
    parse_attention_file,
    plot_attentions,
)

SPACE = "▁"


@dataclass
class Tokenization:
    sentence: str
    spans: List[Tuple[int, int]]

    @staticmethod
    def from_str_tokens(s: str, tokens: List[str]) -> "Tokenization":
        return Tokenization(s, calc_spans(tokens))


def inject_terms(
    src: Tokenization,
    tgt: Tokenization,
    att_matrix: np.ndarray,
    phrase_mapping: Dict[str, str],
    debug=False,
):

    matches = [
        (m.start(), m.end(), l)
        for patt, l in phrase_mapping.items()
        for m in re.finditer(patt, src.sentence)
    ]
    phrases_to_inject = [l for _, _, l in matches]

    match_spans = [(s, e) for s, e, _ in matches]
    src_matches_spans = char_spans_to_token_spans(match_spans, src.spans)

    idx_s2t = att_matrix.argmax(0)
    idx_t2s = att_matrix.argmax(1)

    s2t_n2m = defaultdict(list)
    [insert_or_append(s2t_n2m, s, t) for s, t in enumerate(idx_s2t)]
    [insert_or_append(s2t_n2m, s, t) for t, s in enumerate(idx_t2s)]

    tgt_spans_inject_phrase = [
        (calc_target_span(s, e, s2t_n2m), i)
        for (s, e), i in zip(src_matches_spans, phrases_to_inject)
    ]
    tgt_new = tgt.sentence
    for (s, e), phrase_to_inject in sorted(
        tgt_spans_inject_phrase, key=lambda x: -x[0][1]
    ):
        tgt_new = string_insert(
            tgt_new, tgt.spans[s][0], tgt.spans[e][1], phrase_to_inject,
        )
    src_toks = [src.sentence[s:e] for s, e in src.spans]
    tgt_toks = [tgt.sentence[s:e] for s, e in tgt.spans]
    if debug:
        for s, t in s2t_n2m.items():
            print(f"{src_toks[s]}---{[tgt_toks[k] for k in t]}")

    return tgt_new


def build_words(subwordtokens: List[str]) -> List[str]:
    w2s = calc_word2subword_mapping(subwordtokens)
    words = [
        "".join([subwordtokens[sub_i].replace(SPACE, "") for sub_i in subs])
        for subs in w2s.values()
    ]
    # sentence_str = " ".join(words)
    # s = tokenizer.DecodePieces(subwordtokens)
    # assert s == sentence_str, (s, sentence_str) # just to show that sentencepiece does the same
    return words


def calc_word2subword_mapping(subwords: List[str]) -> Dict[int, List[int]]:
    word2sub = []
    assert subwords[0].startswith(SPACE)
    for k, subwtok in enumerate(subwords):
        if subwtok.startswith(SPACE):
            word2sub.append([k])
        else:
            word2sub[-1].append(k)
    return {i: l for i, l in enumerate(word2sub)}


def map_to_spans(subwords: List[str], words: List[str]):
    word_spans = calc_spans(words)
    subword_spans = calc_spans(subwords)
    return word_spans, subword_spans


def calc_spans(tokens: List[str], debug=False):
    """"
    span ends are exclusive!
    """

    def calc_lens(concat_tok: str):
        return [len(t) + len(concat_tok) for t in tokens[:-1]]

    is_subword = tokens[0].startswith(SPACE)
    if is_subword:
        concat_tok = ""
        lens = calc_lens(concat_tok)
        lens[0] -= 1
    else:
        concat_tok = " "
        lens = calc_lens(concat_tok)

    spans = [(i, i + len(t)) for i, t in zip(numpy.cumsum([0] + lens), tokens)]

    if debug:
        s = concat_tok.join(tokens)
        if is_subword:
            s = s[1:].replace(SPACE, " ")
        print([(s[a:b], t) for (a, b), t in zip(spans, tokens)])

    return spans


def char_spans_to_token_spans(
    char_spans: List[Tuple[int, int]], token_spans: List[Tuple[int, int]]
):
    def find_closest_start_end(char_start, char_end):
        start = int(
            np.argmin(
                [np.abs(token_start - char_start) for token_start, _ in token_spans]
            )
        )
        end = int(
            np.argmin([np.abs(token_end - char_end) for _, token_end in token_spans])
        )
        return start, end

    return [find_closest_start_end(s, e) for s, e in char_spans]


def string_insert(s: str, start: int, end: int, insertion: str):
    return s[:start] + insertion + s[end:]


def calc_target_span(
    s, e, s2t: Dict[int, List[int]]
):  # TODO(tilo): this can go very wrong!
    most_leff = min([t for i in range(s, e + 1) for t in s2t[i]])
    most_right = max([t for i in range(s, e + 1) for t in s2t[i]])
    return (most_leff, most_right)


def insert_or_append(d: Dict, k, v):
    if k in d.keys():
        if v not in d[k]:
            d[k].append(v)
    else:
        d[k] = [v]


def explude_last_and_renormalize(att_matrix, src):
    att_matrix = normalize(att_matrix[:, :-1], axis=1, norm="l2")
    src = src[:-1]
    return att_matrix, src


def renormalize_or_not(
    glossary_file="../glossary.txt",
    attention_file="../sample.attentions",  # attentions file path
):
    with open(glossary_file) as f:
        s2t = dict([l.split("=") for l in f.read().splitlines()])
    data = parse_attention_file(attention_file)
    d = data[0]
    for do_renormalize in [False, True]:
        if do_renormalize:
            att_matrix, src = explude_last_and_renormalize(d.attention, d.src)
        else:
            att_matrix = d.attention
            src = d.src

        tgt = d.tgt
        s = inject_terms(
            Tokenization.from_str_tokens(" ".join(src), src),
            Tokenization.from_str_tokens(" ".join(tgt), tgt),
            att_matrix,
            s2t,
        )
        print(s)
        fig = plot_attentions(tgt=tgt, src=src, att_matrix=att_matrix)
        fig.savefig(f"attention{'_renormalized' if do_renormalize else ''}.png")


def process_attention_file(
    glossary_file="../glossary.txt",
    attention_file="../sample.attentions",
    output_file="processed_phrases.txt",
    debug=False,
):
    data = parse_attention_file(attention_file)
    with open(glossary_file) as f:
        s2t = dict([l.split("=") for l in f.read().splitlines()])
    lines = [
        inject_terms(
            Tokenization.from_str_tokens(" ".join(d.src), d.src),
            Tokenization.from_str_tokens(" ".join(d.tgt), d.tgt),
            d.attention,
            s2t,
        )
        for d in data
    ]
    if debug:
        for l, d in zip(lines, data):
            print((l, " ".join(d.tgt)))

    with open(output_file, "w") as f:
        f.writelines([l + "\n" for l in lines])


if __name__ == "__main__":
    """
    python inject_terminology.py "../glossary.txt" "../sample.attentions" "processed_phrases.txt"
    """
    parser = argparse.ArgumentParser()
    # fmt: off
    parser.add_argument("glossary", type=str)
    parser.add_argument("attention_file", type=str)
    parser.add_argument("output_file",  type=str)
    parser.add_argument("--debug", action="store_true", default=False)
    # fmt: on

    args = parser.parse_args()

    process_attention_file(
        glossary_file=args.glossary,
        attention_file=args.attention_file,
        output_file=args.output_file,
        debug=args.debug,
    )
