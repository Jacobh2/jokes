import pickle
import os
import codecs
import re


_DIGIT_RE = re.compile(r"\d+")

_PAD = "_PAD"
_GO = "_GO"
_EOS = "_EOS"
_UNK = "_UNK"
_START_VOCAB = [_PAD, _GO, _EOS, _UNK]


PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3

START_VOCAB_ID = [PAD_ID, GO_ID, EOS_ID]

def convert_to_id(list_of_sentences, word_to_id):
    list_of_ids = []
    for sentence in list_of_sentences:
        tokens = sentence.split(' ')
        ids = [word_to_id.get(_DIGIT_RE.sub("0", token), UNK_ID) for token in tokens]
        list_of_ids.append(' '.join(list(map(str, ids))))
    return list_of_ids


def load_word_to_id(file_path):
    with open(os.path.join(file_path, 'word_to_id.pkl'), 'rb') as f:
        return pickle.load(f)

def read_data(file_path):

    with codecs.open(os.path.join(file_path, 'questions.txt'), 'r', 'utf-8') as f:
        questions = f.read().split('\n')
    with codecs.open(os.path.join(file_path, 'answers.txt'), 'r', 'utf-8') as f:
        answers = f.read().split('\n')
    with codecs.open(os.path.join(file_path, 'questions_dev.txt'), 'r', 'utf-8') as f:
        questions_dev = f.read().split('\n')
    with codecs.open(os.path.join(file_path, 'answers_dev.txt'), 'r', 'utf-8') as f:
        answers_dev = f.read().split('\n')
    with open(os.path.join(file_path, 'vocab.pkl'), 'rb') as f:
        vocab = pickle.load(f)

    # Make vocab size of 20k
    words = list(map(lambda t: t[0], sorted(vocab.items(), key=lambda i: i[1], reverse=True)))[:20000]

    vocab = []
    vocab.extend(_START_VOCAB)
    vocab.extend(words)

    # create word_to_id
    word_to_id = dict([(w, i) for i, w in enumerate(vocab)])

    assert len(word_to_id) == 20004, "Vocab not 20k"

    q_id = convert_to_id(questions, word_to_id)
    a_id = convert_to_id(answers, word_to_id)
    q_dev_id = convert_to_id(questions_dev, word_to_id)
    a_dev_id = convert_to_id(answers_dev, word_to_id)

    # Save to disc
    q_id_path = os.path.join(file_path, 'questions.txt.ids')
    a_id_path = os.path.join(file_path, 'answers.txt.ids')
    q_id_dev_path = os.path.join(file_path, 'questions_dev.txt.ids')
    a_id_dev_path = os.path.join(file_path, 'answers_dev.txt.ids')

    with codecs.open(q_id_path, 'w', 'utf-8') as f:
        f.write('\n'.join(q_id))
    with codecs.open(a_id_path, 'w', 'utf-8') as f:
        f.write('\n'.join(a_id))
    with codecs.open(q_id_dev_path, 'w', 'utf-8') as f:
        f.write('\n'.join(q_dev_id))
    with codecs.open(a_id_dev_path, 'w', 'utf-8') as f:
        f.write('\n'.join(a_dev_id))

    with open(os.path.join(file_path, 'word_to_id.pkl'), 'wb') as f:
        pickle.dump(word_to_id, f)

    return q_id_path, a_id_path, q_id_dev_path, a_id_dev_path

if __name__ == '__main__':
    print(read_data('data_to_use'))