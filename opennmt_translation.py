from copy import deepcopy

import codecs
import onmt
import os
import re
import torch
from itertools import zip_longest, islice
from onmt.translate import Translator
from onmt.utils import set_random_seed
from onmt.utils.alignment import to_word_align
from onmt.utils.parse import ArgumentParser


def maybe_preprocess(sequence):

    if sequence.get("src", None) is not None:
        sequence = deepcopy(sequence)
        sequence["seg"] = [sequence["src"].strip()]
        sequence.pop("src")
        sequence["ref"] = [sequence.get("ref", None)]
        sequence["n_seg"] = 1

    return sequence


def tokenize(sequence, tokenizer):
    tok = tokenizer.EncodeAsPieces(sequence)
    tok = " ".join(tok)
    return tok


def build_tokenizer(tokenizer_opt, model_root):
    import sentencepiece as spm

    tokenizer = spm.SentencePieceProcessor()
    model_path = os.path.join(model_root, tokenizer_opt["model"])
    tokenizer.Load(model_path)
    return tokenizer


def prepare_data(tokenizers, inputs):
    texts = []
    head_spaces = []
    tail_spaces = []
    all_preprocessed = []
    for i, inp in enumerate(inputs):
        src = inp["src"]
        whitespaces_before, whitespaces_after = "", ""
        match_before = re.search(r"^\s+", src)
        match_after = re.search(r"\s+$", src)
        if match_before is not None:
            whitespaces_before = match_before.group(0)
        if match_after is not None:
            whitespaces_after = match_after.group(0)
        head_spaces.append(whitespaces_before)
        # every segment becomes a dict for flexibility purposes
        seg_dict = maybe_preprocess(inp)
        all_preprocessed.append(seg_dict)
        for seg, ref in zip_longest(seg_dict["seg"], seg_dict["ref"]):
            tok = tokenize(seg, tokenizers["src"])
            if ref is not None:
                ref = tokenize(seg, tokenizers["tgt"])
            texts.append((tok, ref))
        tail_spaces.append(whitespaces_after)
    empty_indices, texts_ref, texts_to_translate = filter_empty_tokens(texts)
    return (
        all_preprocessed,
        empty_indices,
        head_spaces,
        tail_spaces,
        texts_ref,
        texts_to_translate,
    )


def filter_empty_tokens(texts):
    empty_indices = []
    texts_to_translate, texts_ref = [], []
    for i, (tok, ref_tok) in enumerate(texts):
        if tok == "":
            empty_indices.append(i)
        else:
            texts_to_translate.append(tok)
            texts_ref.append(ref_tok)
    if any([item is None for item in texts_ref]):
        texts_ref = None
    return empty_indices, texts_ref, texts_to_translate


def maybe_convert_align(src, tgt, align):
    src_marker = "spacer"
    tgt_marker = "spacer"
    if src_marker is None or tgt_marker is None:
        raise ValueError(
            "To get decoded alignment, joiner/spacer "
            "should be used in both side's tokenizer."
        )
    elif "".join(tgt.split()) != "":
        align = to_word_align(src, tgt, align, src_marker, tgt_marker)
    return align


def maybe_detokenize_with_align(sequence, src, tokenizer, report_align=False):
    align = None
    if report_align:
        # output contain alignment
        sequence, align = sequence.split(" ||| ")
        if align != "":
            align = maybe_convert_align(src, sequence, align)
    sequence = tokenizer.DecodePieces(sequence.split())
    return (sequence, align)


def build_translator(opt,clazz=Translator):
    load_test_model = onmt.model_builder.load_test_model
    fields, model, model_opt = load_test_model(opt)

    scorer = onmt.translate.GNMTGlobalScorer.from_opt(opt)

    translator = clazz.from_opt(
        model,
        fields,
        opt,
        model_opt,
        global_scorer=scorer,
        report_align=opt.report_align,
        report_score=False,
        out_file=codecs.open(os.devnull, "w", "utf-8"),  # WTF!?!?
    )
    return translator


def rebuild_seg_packages(all_preprocessed, results, scores, aligns, n_best):
    offset = 0
    rebuilt_segs = []
    avg_scores = []
    merged_aligns = []
    for i, seg_dict in enumerate(all_preprocessed):
        n_seg = seg_dict["n_seg"]
        sub_results = results[n_best * offset : (offset + n_seg) * n_best]
        sub_scores = scores[n_best * offset : (offset + n_seg) * n_best]
        sub_aligns = aligns[n_best * offset : (offset + n_seg) * n_best]
        for j in range(n_best):
            _seg_dict = deepcopy(seg_dict)
            _seg_dict["seg"] = list(islice(sub_results, j, None, n_best))
            rebuilt_segs.append(_seg_dict)
            sub_sub_scores = list(islice(sub_scores, j, None, n_best))
            avg_score = sum(sub_sub_scores) / n_seg if n_seg != 0 else 0
            avg_scores.append(avg_score)
            sub_sub_aligns = list(islice(sub_aligns, j, None, n_best))
            merged_aligns.append(sub_sub_aligns)
        offset += n_seg
    return rebuilt_segs, avg_scores, merged_aligns


def parse_opt(opt, model_root):

    argv = []
    parser = ArgumentParser()
    onmt.opts.model_opts(parser)
    onmt.opts.translate_opts(parser)

    models = opt["models"]
    if not isinstance(models, (list, tuple)):
        models = [models]
    opt["models"] = [os.path.join(model_root, model) for model in models]
    opt["src"] = "dummy_src"

    for (k, v) in opt.items():
        if k == "models":
            argv += ["-model"]
            argv += [str(model) for model in v]
        elif type(v) == bool:
            argv += ["-%s" % k]
        else:
            argv += ["-%s" % k, str(v)]

    opt = parser.parse_args(argv)
    ArgumentParser.validate_translate_opts(opt)
    opt.cuda = opt.gpu > -1

    return opt


def run_translation(inputs, tokenizers, translator, opt):

    (
        all_preprocessed,
        empty_indices,
        head_spaces,
        tail_spaces,
        texts_ref,
        texts_to_translate,
    ) = prepare_data(tokenizers, inputs)

    assert len(texts_to_translate) > 0
    n_best = opt.n_best
    scores, predictions = translator.translate(
        texts_to_translate,
        tgt=texts_ref,
        batch_size=len(texts_to_translate) if opt.batch_size == 0 else opt.batch_size,
    )

    # NOTE: translator returns lists of `n_best` list
    def flatten_list(_list):
        return sum(_list, [])

    tiled_texts = [t for t in texts_to_translate for _ in range(n_best)]
    results = flatten_list(predictions)

    def maybe_item(x):
        return x.item() if type(x) is torch.Tensor else x

    scores = [maybe_item(score_tensor) for score_tensor in flatten_list(scores)]

    results = [
        maybe_detokenize_with_align(result, src, tokenizers["tgt"], opt.report_align)
        for result, src in zip(results, tiled_texts)
    ]

    aligns = [align for _, align in results]
    results = [tokens for tokens, _ in results]

    # build back results with empty texts
    for i in empty_indices:
        j = i * n_best
        results = results[:j] + [""] * n_best + results[j:]
        aligns = aligns[:j] + [None] * n_best + aligns[j:]
        scores = scores[:j] + [0] * n_best + scores[j:]

    rebuilt_segs, scores, aligns = rebuild_seg_packages(
        all_preprocessed, results, scores, aligns, n_best
    )

    results = [seg["seg"][0] for seg in rebuilt_segs]

    head_spaces = [h for h in head_spaces for i in range(n_best)]
    tail_spaces = [h for h in tail_spaces for i in range(n_best)]
    results = ["".join(items) for items in zip(head_spaces, results, tail_spaces)]

    return results, scores, aligns


if __name__ == "__main__":
    # Translator.calc_avg_attentions = calc_avg_attentions

    model_id = 0
    opt = {
        "gpu": -1,
        "beam_size": 5,
        "models": ["averaged-10-epoch.pt"],
        "report_align": True,
    }
    model_root = "../OpenNMT-py"
    opt = parse_opt(opt, model_root)
    tokenizer_opt = {"type": "sentencepiece", "model": "sentencepiece.model"}

    set_random_seed(opt.seed, opt.cuda)

    tokenizer = build_tokenizer(tokenizer_opt, model_root)
    tokenizers = {"src": tokenizer, "tgt": tokenizer}
    translator = build_translator(opt)

    results, scores, aligns = run_translation(
        [{"src": "is lilt one of lengoo's most feared competitors?", "id": 0}],
        tokenizers=tokenizers,
        translator=translator,
        opt=opt,
    )
    print(results)
    print(aligns)
