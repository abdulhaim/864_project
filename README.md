# 864_project

To run anything in cocoa-master, go to the main directory and set PYTHONPATH to the cocoa-master directory so imports can be found: export PYTHONPATH="$PWD/cocoa-master"

- Following steps here to preprocess the data for end-to-end: https://github.com/abdulhaim/864_project/tree/sl_word/cocoa-master/craigslistbargain
- Before running the following commands, change your working directory to 864_project/cocoa-master/craigslistbargain
1. Build the price tracker 
	- Run with: python3 core/price_tracker.py --train-examples-path data/train.json --output price_tracker
	- outputs to file: cocoa-master/craigslistbargain/price_tracker
2. Build the vocabulary
 	- Run with: python3 main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --mappings mappings/seq2seq --model seq2seq --price-tracker price_tracker --ignore-cache --vocab-only --train-max-examples -1 (Note this is a different command from the README with max-examples -1)
	- in cocoa-master/craigslistbargain/mappings/seq2seq/vocab.pkl
3. GloVe Word Embeddings 
	- filtered on the words in the utterances and product descriptions 
	- in cocoa-master/craigslistbargain/mappings/seq2seq
	- kb_glove = (7087,300)
	- utterance_glove = (10002,300)
	- Can get the utterance embeddings with: torch.load(model_opt.pretrained_wordvec[0]) 
        - Can get word to index mappings with: mappings["src_vocab"].word_to_ind (https://github.com/abdulhaim/864_project/blob/9de695ef28d8610eb88a23ecc13a8933c6bb9426/cocoa-master/cocoa/model/vocab.py)
4. Full Seq2Seq Model
	- Rename either the Word2Vec or Glove embeddings in mappings/seq2seq to utterance_glove.pt and kb_glove.pt depending on which you want to use
	- Run with: python3 main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --test-examples-paths data/dev.json --price-tracker price_tracker --model seq2seq --model-path checkpoint/seq2seq --mappings mappings/seq2seq --pretrained-wordvec mappings/seq2seq/utterance_glove.pt mappings/seq2seq/kb_glove.pt --word-vec-size 300 --rnn-size 300 --rnn-type LSTM --global-attention multibank_general --enc-layers 2 --dec-layers 2 --num-context 2 --batch-size 128 --optim adagrad --learning-rate 0.01  --report-every 500 --epochs 15 --cache cache/seq2seq --ignore-cache --verbose --train-max-examples -1 --test-max-examples -1
5. Full LF2LF model (sl_act)
	- Run with: python main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train-parsed.json --test-examples-paths data/dev-parsed.json --price-tracker price_tracker.pkl --model lf2lf --model-path checkpoint/lf2lf --mappings mappings/lf2lf --word-vec-size 300 --pretrained-wordvec '' '' --rnn-size 300 --rnn-type LSTM --global-attention multibank_general --num-context 2 --stateful --batch-size 128 --optim adagrad --learning-rate 0.01 --epochs 15 --report-every 500 --cache cache/lf2lf --ignore-cache --verbose --train-max-examples -1 --test-max-examples -1
