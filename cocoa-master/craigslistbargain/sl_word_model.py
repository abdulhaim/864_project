import torch.nn as nn

import onmt
import onmt.io
import onmt.Models
import onmt.modules
from cocoa.core.schema import Schema
from neural import get_data_generator
from neural.models import NegotiationModel

def make_model(opt, mappings):
    """
    Makes a negotiation model for seq2seq (sl(word)) architecture
    """
    # make generic embeddings
    src_embeddings = make_embeddings(opt, mappings['src_vocab'], True)
    tgt_embeddings = make_embeddings(opt, mappings['tgt_vocab'], False)

    # make context embedder.
    if opt.num_context > 0:
        context_dict = mappings['utterance_vocab']
        context_embeddings = make_embeddings(opt, context_dict)
        context_embedder = onmt.MeanEncoder(opt.enc_layers, context_embeddings, 'utterance')


    # Make kb embedder.
    if "multibank" in opt.global_attention:
        if opt.model == 'lf2lf':
            kb_embedder = None
        else:
            kb_dict = mappings['kb_vocab']
            kb_embeddings = make_embeddings(opt, kb_dict)
            kb_embedder = onmt.MeanEncoder(opt.enc_layers, kb_embeddings, 'kb')
    
    # make encoder and decoder with generic embeddings
    encoder = onmt.encoders.TransformerEncoder(opt.num_layers, 
                                               opt.rnn_size, 
                                               opt.heads, 
                                               opt.hidden_size, 
                                               opt.dropout, 
                                               src_embeddings)
    decoder = onmt.decoders.TransformerDecoder(opt.num_layers, 
                                               opt.rnn_size, 
                                               opt.global_attention, 
                                               opt.copy_attn,
                                               opt.dropout,
                                               tgt_embeddings
                                               )

    # incorporate the pretrained vectors into the model                                           
    model = NegotiationModel(encoder, decoder, context_embedder, kb_embedder)
    model.encoder.embeddings.load_pretrained_vectors(opt.pretrained_wordvec[0], False)
    model.context_embedder.embeddings.load_pretrained_vectors(opt.pretrained_wordvec[0], False)
    model.kb_embedder.embeddings.load_pretrained_vectors(opt.pretrained_wordvec[1], False)
    model.decoder.embeddings.load_pretrained_vectors(opt.pretrained_wordvec[0], False)
    return model


def make_embeddings(opt, word_dict, for_encoder=True):
    """
    Make an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocabulary): words dictionary.
        for_encoder(bool): make Embeddings for encoder or decoder?
    """
    embedding_dim = opt.word_vec_size

    word_padding_idx = word_dict.to_ind(markers.PAD)
    num_word_embeddings = len(word_dict)

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=False,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      word_vocab_size=num_word_embeddings)
    
def main():
    parser = argparse.ArgumentParser()
    model_args = parser.parse_args()
    schema = Schema(model_args.schema_path, None)
    data_generator = get_data_generator(model_args, model_args, schema)
    mappings = data_generator.mappings

    model = make_model(model_args, mappings)

   
   # To run the code, use these arguments
    # PYTHONPATH=. python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev-parsed.json \
    # --price-tracker price_tracker.pkl \
    # --model seq2seq \
    # --model-path checkpoint/seq2seq --mappings mappings/seq2seq \
    # --pretrained-wordvec mappings/seq2seq/utterance_glove.pt mappings/seq2seq/kb_glove.pt --word-vec-size 300 \
    # --rnn-size 300 --rnn-type LSTM --global-attention multibank_general \
    # --enc-layers 2 --dec-layers 2 --num-context 2 \
    # --batch-size 128 --gpuid 0 --optim adagrad --learning-rate 0.01  \
    # --report-every 500 \
    # --epochs 15 \
    # --cache cache/seq2seq --ignore-cache \
    # --verbose

if __name__ == '__main__':
    main()
