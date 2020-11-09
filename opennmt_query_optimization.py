import numpy
import onmt
import torch
from onmt.models import NMTModel
from onmt.modules import MultiHeadedAttention, AverageAttention
from onmt.translate import Translator, Translation
from onmt.translate.translator import max_tok_len
from sklearn.preprocessing import normalize
from torch.nn import Parameter
from tqdm import tqdm
from types import MethodType
from typing import Tuple, Any

from opennmt_lexicon_injection import prepare_opt_tokenizers, build_data_iter
from opennmt_translation import build_translator, prepare_data
from plot_attentions import plot_attentions


def _forward_override_learnable_query(
    self,
    inputs,
    memory_bank,
    src_pad_mask,
    tgt_pad_mask,
    layer_cache=None,
    step=None,
    future=False,
):
    """
    based on: https://github.com/OpenNMT/OpenNMT-py/blob/bb2d045f866a40557c3753b0a1be1bcb7fd1866e/onmt/decoders/transformer.py#L14
    A naive forward pass for transformer decoder.

    # T: could be 1 in the case of stepwise decoding or tgt_len

    Args:
        inputs (FloatTensor): ``(batch_size, T, model_dim)``
        memory_bank (FloatTensor): ``(batch_size, src_len, model_dim)``
        src_pad_mask (bool): ``(batch_size, 1, src_len)``
        tgt_pad_mask (bool): ``(batch_size, 1, T)``
        layer_cache (dict or None): cached layer info when stepwise decode
        step (int or None): stepwise decoding counter
        future (bool): If set True, do not apply future_mask.

    Returns:
        (FloatTensor, FloatTensor):

        * output ``(batch_size, T, model_dim)``
        * attns ``(batch_size, head, T, src_len)``

    """

    very_first_forward_pass = not hasattr(self, "query_param")
    if very_first_forward_pass:
        dec_mask = _build_decoder_mask(future, step, tgt_pad_mask)
        _initialize_query_param(self, dec_mask, inputs, layer_cache, step)

    query = self.query_param
    query_norm = self.layer_norm_2(query)
    mid, attns = self.context_attn(
        memory_bank,
        memory_bank,
        query_norm,
        mask=src_pad_mask,
        layer_cache=layer_cache,
        attn_type="context",
    )
    output = self.feed_forward(self.drop(mid) + query)

    return output, attns


def _initialize_query_param(self, dec_mask, inputs, layer_cache, step):
    input_norm = self.layer_norm_1(inputs)
    if isinstance(self.self_attn, MultiHeadedAttention):
        query, _ = self.self_attn(
            input_norm,
            input_norm,
            input_norm,
            mask=dec_mask,
            layer_cache=layer_cache,
            attn_type="self",
        )
    elif isinstance(self.self_attn, AverageAttention):
        query, _ = self.self_attn(
            input_norm, mask=dec_mask, layer_cache=layer_cache, step=step
        )
    query = self.drop(query) + inputs
    self.query_param = Parameter(torch.Tensor(query.shape))
    self.query_param.data.copy_(query.data)
    self.query_param.requires_grad = True


def _build_decoder_mask(future, step, tgt_pad_mask):
    dec_mask = None
    if step is None:
        tgt_len = tgt_pad_mask.size(-1)
        if not future:  # apply future_mask, result mask in (B, T, T)
            future_mask = torch.ones(
                [tgt_len, tgt_len], device=tgt_pad_mask.device, dtype=torch.uint8
            )
            future_mask = future_mask.triu_(1).view(1, tgt_len, tgt_len)
            # BoolTensor was introduced in pytorch 1.2
            try:
                future_mask = future_mask.bool()
            except AttributeError:
                pass
            dec_mask = torch.gt(tgt_pad_mask + future_mask, 0)
        else:  # only mask padding, result mask in (B, 1, T)
            dec_mask = tgt_pad_mask
    return dec_mask


import torch.optim as optim
import torch.nn as nn


def train(model: NMTModel, num_epochs, fields, src, src_reader, tgt, tgt_reader):
    data, data_iter = build_data_iter(src, tgt, fields, src_reader, tgt_reader, 1,)
    batch = next(iter(data_iter))
    src_tensor, src_len = batch.src
    outputs, attns = model(src_tensor, batch.tgt, src_len)
    optimizer = optim.SGD([p for p in model.parameters() if p.requires_grad], lr=10.0)
    tgt_field = dict(fields)["tgt"].base_field
    padding_idx = tgt_field.vocab.stoi[tgt_field.pad_token]
    criterion = nn.NLLLoss(ignore_index=padding_idx, reduction="sum")

    def flatten(_v):
        return _v.view(-1, _v.size(2))

    for k in tqdm(range(num_epochs)):
        optimizer.zero_grad()

        bottled_output = flatten(outputs)
        target = batch.tgt[1:, :, 0]
        gtruth = target.view(-1)

        scores = model.generator(bottled_output)

        loss = criterion(scores, gtruth)

        loss.backward()
        optimizer.step()
        outputs, attns = model(src_tensor, batch.tgt, src_len)


class QueryOptimizingTranslator(Translator):
    def translate_optimize_query(self, src, num_epochs: int) -> Tuple[Translation, Any]:

        src_reader = self.src_reader
        tgt_reader = self.tgt_reader
        fields = self.fields

        trans = self.translate_single_phrase(src, fields, src_reader)

        tgt = [" ".join(trans.pred_sents[0])]

        self.make_decoder_layers_query_learnable()
        if num_epochs > 0:
            train(self.model, num_epochs, fields, src, src_reader, tgt, tgt_reader)

        align = self.calc_alignment(src, tgt, fields, src_reader, tgt_reader)
        return trans, align

    def make_decoder_layers_query_learnable(self):
        layer_idx = translator.model.decoder.alignment_layer
        decoder_layer = translator.model.decoder.transformer_layers[layer_idx]
        decoder_layer._forward = MethodType(
            _forward_override_learnable_query, decoder_layer
        )

    def translate_single_phrase(self, src, fields, src_reader):
        data, data_iter = build_data_iter(src, None, fields, src_reader, None, 1,)
        xlation_builder = onmt.translate.TranslationBuilder(
            data, self.fields, self.n_best, self.replace_unk, None, self.phrase_table
        )
        batch = next(iter(data_iter))
        batch_data = self.translate_batch(batch, data.src_vocabs, attn_debug=False)
        translations = xlation_builder.from_batch(batch_data)
        trans = translations[0]
        return trans

    def calc_alignment(self, src, tgt, fields, src_reader, tgt_reader):
        data, data_iter = build_data_iter(src, tgt, fields, src_reader, tgt_reader, 1,)
        batch = next(iter(data_iter))
        target = batch.tgt[1:, :, 0]
        target = target.view(-1)
        align = self._align_forward(
            batch, [[target]]
        )  # TODO(tilo): what is this method actually doing?
        align = align[0][0].detach().numpy()
        return align


def plot_save(file_name, att_matrix, src):
    fig = plot_attentions(
        tgt=tgt,
        src=src[: -len(parking_position)],
        att_matrix=att_matrix,
        figsize=(12, 12),
    )
    fig.savefig(file_name)


if __name__ == "__main__":
    opt, tokenizers = prepare_opt_tokenizers()
    opt.train_from = None
    translator = build_translator(opt, clazz=QueryOptimizingTranslator)

    alignment_layer = -3
    translator.model.decoder.alignment_layer = alignment_layer  # WTF -> only necessary cause OpenNMT does obscure argument parsing!

    parking_position = (
        " ."  # why does it not accept eos = tokenizer.IdToPiece(tokenizer.eos_id())
    )
    # src_text = "is lilt one of lengoo's most feared competitors?"
    src_text = "The resulting alignments dramatically outperform the naive approach to interpreting Transformer attention activations."  # Zenkel2019
    inputs = [{"src": (f"{src_text}{parking_position}"), "id": 0,}]
    (
        all_preprocessed,
        empty_indices,
        head_spaces,
        tail_spaces,
        texts_ref,
        texts_to_translate,
    ) = prepare_data(tokenizers, inputs)

    assert len(texts_to_translate) > 0

    for par in translator.model.parameters():
        par.requires_grad = False

    num_epochs = 100
    trans, att_matrix = translator.translate_optimize_query(
        texts_to_translate, num_epochs=num_epochs,
    )
    att_matrix_original = trans.word_aligns[0].numpy()
    print(
        f"attention-matrizes are differing by {numpy.linalg.norm(att_matrix-att_matrix_original):.4f}"
    )
    src = trans.src_raw
    tgt = trans.pred_sents[0]
    att_orig_norm = normalize(
        att_matrix_original[:, : -len(parking_position)], axis=1, norm="l2"
    )
    plot_save(f"attention_iter_original.png", att_orig_norm, src)
    att_optimized_norm = normalize(
        att_matrix[:, : -len(parking_position)], axis=1, norm="l2"
    )
    plot_save(
        f"attention_differnce_iter_{num_epochs}.png",
        att_optimized_norm - att_orig_norm,
        src,
    )
