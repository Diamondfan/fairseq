# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
from argparse import Namespace
from typing import Any

import torch
import torch.nn as nn
from dataclasses import dataclass, field
from fairseq import checkpoint_utils, tasks, utils
from fairseq.dataclass import FairseqDataclass
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from fairseq.models import BaseFairseqModel, FairseqEncoder, register_model
from fairseq.models.hubert.hubert import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import LayerNorm
from fairseq.tasks import FairseqTask
from omegaconf import II, MISSING


@dataclass
class HubertAsrConfig(FairseqDataclass):
    w2v_path: str = field(default=MISSING, metadata={"help": "path to hubert model"})
    no_pretrained_weights: bool = field(
        default=False,
        metadata={"help": "if true, does not load pretrained weights"},
    )
    bottleneck_dim: int = field(
        default=0, metadata={"help": "bottleneck dimension for residual adapter"}
    )
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    final_dropout: float = field(
        default=0.0,
        metadata={"help": "dropout after transformer and before final projection"},
    )
    dropout: float = field(
        default=0.0,
        metadata={"help": "dropout probability inside hubert model"},
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside hubert model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside hubert model"
        },
    )
    freeze_adapter: bool = field(
        default=False,
        metadata={"help": "freeze paramters in adapters or not"},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={"help": "freeze backbone parameters or not"},
    )
    use_first_adapter: bool = field(
        default=True,
        metadata={"help": "the position of adapter"},
    )
    adapter_before_quant: bool = field(
        default=False,
        metadata={"help": "the position of adapter"},
    )

    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask "
            "(normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )

    # channel masking
    mask_channel_length: int = field(
        default=10,
        metadata={"help": "length of the mask for features (channels)"},
    )
    mask_channel_prob: float = field(
        default=0.0,
        metadata={"help": "probability of replacing a feature with 0"},
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument "
            "(used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False,
        metadata={"help": "whether to allow channel masks to overlap"},
    )
    freeze_finetune_updates: int = field(
        default=0,
        metadata={"help": "dont finetune hubert for this many updates"},
    )
    feature_grad_mult: float = field(
        default=0.0,
        metadata={"help": "reset feature grad mult in hubert to this"},
    )
    layerdrop: float = field(
        default=0.0,
        metadata={"help": "probability of dropping a layer in hubert"},
    )
    normalize: bool = II("task.normalize")
    data: str = II("task.data")

    # this holds the loaded hubert args
    w2v_args: Any = None

    final_layer_type: str = field(
        default="Linear",
        metadata={"help":"final layer type"},
    )
    
    lstm_dim: int = field(
        default=1024, metadata={"help": "lstm dimension for final layer"}
    )
    lstm_num_layer: int = field(
        default=2, metadata={"help": "number of layers of final LSTM module"}
    )
    lstm_bidirection: bool = field(
        default=True, metadata={"help":"bidirectional LSTM or not"}
    )
    lstm_module: str = field(
        default="lstm", metadata={"help": "rnn type, lstm, rnn, gru"}
    )
    lstm_add_norm: bool = field(
        default=False, metadata={"help": "use layer norm in final layer or not"}
    )
    lstm_add_prob: bool = field(
        default=False, metadata={"help": "add proj in final lstm module or not"}
    )
    lstm_dropout: float = field(
        default=0.2, metadata={"help": "dropout in lstm module"}
    )

@dataclass
class HubertCtcConfig(HubertAsrConfig):
    pass


@register_model("hubert_ctc", dataclass=HubertCtcConfig)
class HubertCtc(BaseFairseqModel):
    def __init__(self, cfg: HubertCtcConfig, w2v_encoder: BaseFairseqModel):
        super().__init__()
        self.cfg = cfg
        self.w2v_encoder = w2v_encoder

    def upgrade_state_dict_named(self, state_dict, name):
        super().upgrade_state_dict_named(state_dict, name)
        return state_dict

    @classmethod
    def build_model(cls, cfg: HubertCtcConfig, task: FairseqTask):
        """Build a new model instance."""
        w2v_encoder = HubertEncoder(cfg, task.target_dictionary)
        return cls(cfg, w2v_encoder)

    def get_normalized_probs(self, net_output, log_probs):
        #import pdb
        #pdb.set_trace()
        """Get normalized probabilities (or log probs) from a net's output."""

        logits = self.get_logits(net_output) #net_output["encoder_out"]
        if log_probs:
            return utils.log_softmax(logits.float(), dim=-1)
        else:
            return utils.softmax(logits.float(), dim=-1)

    def get_logits(self, net_output):
        logits = net_output["encoder_out"]
        padding = net_output["encoder_padding_mask"]
        if padding is not None and padding.any():
            number_of_classes = logits.size(-1)
            masking_tensor = torch.ones(
                number_of_classes, device=logits.device
            ) * float("-inf")
            masking_tensor[0] = 0
            logits[padding.T] = masking_tensor.type_as(logits)
            #padding = padding.T
            #logits[padding][..., 0] = 0
            #logits[padding][..., 1:] = float("-inf")

        return logits

    def forward(self, **kwargs):
        x = self.w2v_encoder(**kwargs)
        return x


@dataclass
class HubertSeq2SeqConfig(HubertAsrConfig):
    decoder_embed_dim: int = field(
        default=768, metadata={"help": "decoder embedding dimension"}
    )
    decoder_ffn_embed_dim: int = field(
        default=3072, metadata={"help": "decoder embedding dimension for FFN"}
    )
    decoder_layers: int = field(default=6, metadata={"help": "num of decoder layers"})
    decoder_layerdrop: float = field(
        default=0.0, metadata={"help": "decoder layerdrop chance"}
    )
    decoder_attention_heads: int = field(
        default=4, metadata={"help": "num decoder attention heads"}
    )
    decoder_learned_pos: bool = field(
        default=False,
        metadata={"help": "use learned positional embeddings in the decoder"},
    )
    decoder_normalize_before: bool = field(
        default=False,
        metadata={"help": "apply layernorm before each decoder block"},
    )
    no_token_positional_embeddings: bool = field(
        default=False,
        metadata={
            "help": "if set, disables positional embeddings " "(outside self attention)"
        },
    )
    decoder_dropout: float = field(
        default=0.0, metadata={"help": "dropout probability in the decoder"}
    )
    decoder_attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights " "inside the decoder"
        },
    )
    decoder_activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN " "inside the decoder"
        },
    )
    max_target_positions: int = field(
        default=2048, metadata={"help": "max target positions"}
    )
    share_decoder_input_output_embed: bool = field(
        default=False,
        metadata={"help": "share decoder input and output embeddings"},
    )

class RNNLayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, module, bidirection, dropout, add_proj=False, add_norm=False):
        super(RNNLayer, self).__init__()
        self.layer = getattr(nn, module.upper())(input_dim, hidden_dim, bidirectional=bidirection, num_layers=1, batch_first=False)
        out_dim = 2 * hidden_dim if bidirection else hidden_dim

        self.out_dim = out_dim
        self.add_norm = add_norm
        self.add_proj = add_proj

        if add_norm:
            self.norm = LayerNorm(out_dim)
        self.dropout = nn.Dropout(p=dropout)

        if self.add_proj:
            self.proj = nn.Linear(out_dim, out_dim)

    def forward(self, x): 
        x, _ = self.layer(x)
    
        if self.add_norm:
            x = self.norm(x)
    
        x = self.dropout(x)

        if self.add_proj:
            x = self.proj(x)

        return x

class LSTMGenerator(nn.Module):
    "Using SSL as feature extractor"
    def __init__(self, d_input, vocab, cfg):
        super(LSTMGenerator, self).__init__()
        rnn_input = d_input
        d_lstm = cfg.lstm_dim
        num_layer = cfg.lstm_num_layer
        bidirection = cfg.lstm_bidirection
        module = cfg.lstm_module
        add_norm = cfg.lstm_add_norm
        add_proj = cfg.lstm_add_prob
        dropout = cfg.lstm_dropout

        self.layers = nn.ModuleList()
        for i in range(num_layer):
            rnn_layer = RNNLayer(rnn_input, d_lstm, module, bidirection, dropout, add_proj, add_norm)
            self.layers.append(rnn_layer)
            rnn_input = rnn_layer.out_dim

        self.final_layer = nn.Linear(rnn_input, vocab)

    def forward(self, x, T=1.0):
        for layer in self.layers:
            x = layer(x)
        output = self.final_layer(x)
        return output


class HubertEncoder(FairseqEncoder):
    def __init__(self, cfg: HubertAsrConfig, tgt_dict=None):
        self.apply_mask = cfg.apply_mask

        arg_overrides = {
            "dropout": cfg.dropout,
            "activation_dropout": cfg.activation_dropout,
            "dropout_input": cfg.dropout_input,
            "attention_dropout": cfg.attention_dropout,
            "mask_length": cfg.mask_length,
            "mask_prob": cfg.mask_prob,
            "mask_selection": cfg.mask_selection,
            "mask_other": cfg.mask_other,
            "no_mask_overlap": cfg.no_mask_overlap,
            "mask_channel_length": cfg.mask_channel_length,
            "mask_channel_prob": cfg.mask_channel_prob,
            "mask_channel_selection": cfg.mask_channel_selection,
            "mask_channel_other": cfg.mask_channel_other,
            "no_mask_channel_overlap": cfg.no_mask_channel_overlap,
            "encoder_layerdrop": cfg.layerdrop,
            "feature_grad_mult": cfg.feature_grad_mult,
            "bottleneck_dim": cfg.bottleneck_dim,
            "freeze_adapter": cfg.freeze_adapter,
            "freeze_backbone": cfg.freeze_backbone,
            "use_first_adapter": cfg.use_first_adapter,
            "adapter_before_quant": cfg.adapter_before_quant,
        }

        if cfg.w2v_args is None:
            state = checkpoint_utils.load_checkpoint_to_cpu(cfg.w2v_path, arg_overrides)
            w2v_args = state.get("cfg", None)
            if w2v_args is None:
                w2v_args = convert_namespace_to_omegaconf(state["args"])
            cfg.w2v_args = w2v_args
        else:
            state = None
            w2v_args = cfg.w2v_args
            if isinstance(w2v_args, Namespace):
                cfg.w2v_args = w2v_args = convert_namespace_to_omegaconf(w2v_args)

        assert cfg.normalize == w2v_args.task.normalize, (
            "Fine-tuning works best when data normalization is the same. "
            "Please check that --normalize is set or unset for "
            "both pre-training and here"
        )

        w2v_args.task.data = cfg.data
        task = tasks.setup_task(w2v_args.task)
        if state is not None and "task_state" in state:
            # This will load the stored "dictionaries" object
            task.load_state_dict(state["task_state"])
        w2v_args.model.no_pretrained_weights = True
        model = task.build_model(w2v_args.model)

        if state is not None and not cfg.no_pretrained_weights:
            print("load pretrained hubert model from {}".format(cfg.w2v_path))
            # set strict=False because we omit some modules
            model.load_state_dict(state["model"], False)

        model.remove_pretraining_modules()

        super().__init__(task.source_dictionary)

        d = w2v_args.model.encoder_embed_dim

        self.w2v_model = model

        self.final_dropout = nn.Dropout(cfg.final_dropout)
        self.freeze_finetune_updates = cfg.freeze_finetune_updates
        self.num_updates = 0

        if tgt_dict is not None:
            targ_d = len(tgt_dict)
            #self.proj = Linear(d, len(tgt_dict))
        elif getattr(cfg, "decoder_embed_dim", d) != d:
            targ_d = cfg.decoder_embed_dim
            #self.proj = Linear(d, cfg.decoder_embed_dim)
        else:
            targ_d = None
            #self.proj = None

        if targ_d is not None:
            if cfg.final_layer_type == "Linear":
                self.proj = Linear(d, targ_d)
            elif cfg.final_layer_type == "LSTM":
                self.proj = LSTMGenerator(d, targ_d, cfg)
            else:
                raise NotImplementedError
        else:
            self.proj = None

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates

    def forward(self, source, padding_mask, tbc=True, **kwargs):

        w2v_args = {
            "source": source,
            "padding_mask": padding_mask,
            "mask": self.apply_mask and self.training,
        }

        ft = self.freeze_finetune_updates <= self.num_updates

        with torch.no_grad() if not ft else contextlib.ExitStack():
            x, padding_mask = self.w2v_model.extract_features(**w2v_args)

            if tbc:
                # B x T x C -> T x B x C
                x = x.transpose(0, 1)

        x = self.final_dropout(x)

        if self.proj:
            x = self.proj(x)

        return {
            "encoder_out": x,  # T x B x C
            "encoder_padding_mask": padding_mask,  # B x T
            "padding_mask": padding_mask,
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        if encoder_out["encoder_out"] is not None:
            encoder_out["encoder_out"] = encoder_out["encoder_out"].index_select(
                1, new_order
            )
        if encoder_out["encoder_padding_mask"] is not None:
            encoder_out["encoder_padding_mask"] = encoder_out[
                "encoder_padding_mask"
            ].index_select(0, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return None

    def upgrade_state_dict_named(self, state_dict, name):
        return state_dict


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.0)
    return m
