The processed and normalized data can be found in
- `answers.txt`
- `questions.txt`

These are in human readable form. For the dev/valid part, check out
- `answers_dev.txt`
- `questions_dev.txt`

Then there is versions for the network: converted into IDs called `*.ids`.

`vocab.pkl` is a pickled list of all words in the vocabulary and the `word_to_id.pkl` is a pickled dict mapping from word to unique ID.

All these was generated from the datasource files found in the `raw_data` folder using the `process.py` script under `preprocessing`