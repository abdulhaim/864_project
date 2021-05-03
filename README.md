# 864_project

To run anything in cocoa-master, set PYTHONPATH to the cocoa-master directory so imports can be found: export PYTHONPATH="$PWD/cocoa-master"

-Following steps here to preprocess the data: https://github.com/abdulhaim/864_project/tree/sl_word/cocoa-master/craigslistbargain

1. Build the price tracker-DONE (is the file cocoa-master/craigslistbargain/price_tracker)
2. Build the vocabulary-DONE -vocab in cocoa-master/craigslistbargain/mappings/seq2seq/vocab.pkl
	- Ran with python3 main.py --schema-path data/craigslist-schema.json --train-examples-paths data/train.json --mappings mappings/seq2seq --model seq2seq --price-tracker price_tracker --ignore-cache --vocab-only --train-max-examples -1 (Note this is a different command from the README with max-examples -1)