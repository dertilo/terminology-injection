from typing import Generator, List

import onmt
from onmt import inputters
from onmt.translate import Translator, Translation
from onmt.translate.translator import max_tok_len
from onmt.utils import set_random_seed
from sklearn.preprocessing import normalize

from glossary_attention.read_plot_attentions import plot_attentions
from opennmt_translation import (
    prepare_data,
    parse_opt,
    build_tokenizer,
    build_translator,
)
from glossary_attention.inject_terminology import (
    inject_terms,
    build_words,
    map_to_spans,
    Tokenization,
)


def build_data_iter(
    src,
    tgt,
    fields,
    src_reader,
    tgt_reader,
    batch_size,
    batch_type="sents",
    data_type="text",
    device="cpu",
    filter_pred=None,
):
    src_data = {"reader": src_reader, "data": src, "dir": None}
    tgt_data = {"reader": tgt_reader, "data": tgt, "dir": None}
    _readers, _data, _dir = inputters.Dataset.config(
        [("src", src_data), ("tgt", tgt_data)]
    )
    # corpus_id field is useless here
    if fields.get("corpus_id", None) is not None:
        fields.pop("corpus_id")
    data = inputters.Dataset(
        fields,
        readers=_readers,
        data=_data,
        dirs=_dir,
        sort_key=inputters.str2sortkey[data_type],
        filter_pred=filter_pred,
    )
    data_iter = inputters.OrderedIterator(
        dataset=data,
        device=device,
        batch_size=batch_size,
        batch_size_fn=max_tok_len if batch_type == "tokens" else None,
        train=False,
        sort=False,
        sort_within_batch=True,
        shuffle=False,
    )
    return data, data_iter


class TilosTranslator(Translator):
    def translation_generator(
        self, src, tgt=None, batch_size=None,
    ) -> Generator[Translation, None, None]:

        src_reader = self.src_reader
        tgt_reader = self.tgt_reader
        fields = self.fields
        # data_type = self.data_type
        # filter_pred = self._filter_pred
        # device = self._dev

        data, data_iter = build_data_iter(
            src, tgt, fields, src_reader, tgt_reader, batch_size,
        )

        xlation_builder = onmt.translate.TranslationBuilder(
            data, fields, self.n_best, self.replace_unk, tgt, self.phrase_table
        )

        for batch in data_iter:
            batch_data = self.translate_batch(batch, data.src_vocabs, attn_debug=False)
            translations = xlation_builder.from_batch(batch_data)

            for trans in translations:
                yield trans


def build_tokenizers(model_root: str):
    tokenizer_opt = {"type": "sentencepiece", "model": "sentencepiece.model"}
    tokenizer = build_tokenizer(tokenizer_opt, model_root)
    tokenizers = {"src": tokenizer, "tgt": tokenizer}
    return tokenizers


def prepare_opt_tokenizers():
    opt = {
        "gpu": -1,
        "beam_size": 5,
        "models": ["averaged-10-epoch.pt"],
        "report_align": True,
    }
    model_root = "../OpenNMT-py"
    opt = parse_opt(opt, model_root)
    set_random_seed(opt.seed, opt.cuda)
    tokenizers = build_tokenizers(model_root)
    return opt, tokenizers


def string_and_token_spans(
    src_subw: List[str], tgt_subw: List[str],
):
    src_words = build_words(src_subw)
    src_word_spans, src_subword_spans = map_to_spans(src_subw, src_words)
    tgt_words = build_words(tgt_subw)
    tgt_word_spans, tgt_subword_spans = map_to_spans(tgt_subw, tgt_words)
    src_s = " ".join(src_words)
    tgt_s = " ".join(tgt_words)
    return src_s, src_subword_spans, tgt_s, tgt_subword_spans


if __name__ == "__main__":

    opt, tokenizers = prepare_opt_tokenizers()
    translator = build_translator(opt, clazz=TilosTranslator)
    alignment_layer = -3
    translator.model.decoder.alignment_layer = alignment_layer  # WTF -> only necessary cause OpenNMT does obscure argument parsing!

    parking_position = (
        " ."  # why does it not accept eos = tokenizer.IdToPiece(tokenizer.eos_id())
    )
    inputs = [
        {
            "src": (
                "is lilt one of lengoo's most feared competitors?%s" % parking_position
            ),
            "id": 0,
        }
    ]
    (
        all_preprocessed,
        empty_indices,
        head_spaces,
        tail_spaces,
        texts_ref,
        texts_to_translate,
    ) = prepare_data(tokenizers, inputs)

    assert len(texts_to_translate) > 0
    g = translator.translation_generator(
        texts_to_translate,
        tgt=texts_ref,
        batch_size=len(texts_to_translate) if opt.batch_size == 0 else opt.batch_size,
    )
    trans = next(g)

    phrase_mapping = {
        r"(?:(?:L|l)ilt|LILT)": "TheOneWhoMustNotBeNamed",
        r"most feared": "am meisten gehasst",
        r"competitors?": "Mitstreiter",
    }
    src = trans.src_raw
    tgt = trans.pred_sents[0]
    att_matrix = trans.word_aligns[0].numpy()
    att_matrix = normalize(att_matrix[:, : -len(parking_position)], axis=1, norm="l2")
    src = src[: -len(parking_position)]

    src_s, src_subword_spans, tgt_s, tgt_subword_spans = string_and_token_spans(
        src, tgt
    )

    tgt_substituted = inject_terms(
        Tokenization(src_s, src_subword_spans),
        Tokenization(tgt_s, tgt_subword_spans),
        att_matrix,
        phrase_mapping,
    )

    fig = plot_attentions(tgt=tgt, src=src, att_matrix=att_matrix)
    fig.savefig(f"renormalized_without_pp_{alignment_layer}.png")
