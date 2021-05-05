import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules


def make_model(opt, embeddings):
    encoder = onmt.encoders.TransformerEncoder(opt['num_layers'], 
                                               opt['d_model'], 
                                               opt['heads'], 
                                               opt['hidden_size'], 
                                               opt['dropout'], 
                                               embeddings)
    decoder = onmt.decoders.TransformerDecoder(opt['num_layers'], 
                                               opt['d_model'], 
                                               opt['heads'], 
                                               opt['hidden_size'], 
                                               False,
                                               'scaled_dot'
                                               opt['dropout'],
                                               opt['dropout'] # TODO attention dropout
                                               )
    model = onmt.NMTModel(encoder, decoder)
    return model

    # Args:
    #     num_layers (int): number of decoder layers.
    #     d_model (int): size of the model
    #     heads (int): number of heads
    #     d_ff (int): size of the inner FF layer
    #     copy_attn (bool): if using a separate copy attention
    #     self_attn_type (str): type of self-attention scaled-dot, average
    #     dropout (float): dropout in residual, self-attn(dot) and feed-forward
    #     attention_dropout (float): dropout in context_attn (and self-attn(avg))
    #     embeddings (onmt.modules.Embeddings):
    #         embeddings to use, should have positional encodings
    #     max_relative_positions (int):
    #         Max distance between inputs in relative positions representations
    #     aan_useffn (bool): Turn on the FFN layer in the AAN decoder
    #     full_context_alignment (bool):
    #         whether enable an extra full context decoder forward for alignment
    #     alignment_layer (int): N° Layer to supervise with for alignment guiding
    #     alignment_heads (int):
    #         N. of cross attention heads to use for alignment guiding

def main():
    opt = {'num_layers': 2,
            'd_model': 300,
            'hidden_size': 300, #same as d_ff
            'heads': 8,
            'dropout': .5
            }
    embeddings = # code for getting embeddings from GloVe pretraining
    model = make_model(opt, embeddings)