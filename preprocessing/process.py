import pandas
import os
from nltk.tokenize import word_tokenize
import pickle
from collections import Counter
import codecs


# Read all data files
def data():
    for file_name in os.listdir('jokes'):
        path = "jokes/{}".format(file_name)
        print("Loading", path)
        df = pandas.read_csv(path)
        print("Path loaded!")
        qs = df['Question'].values
        aas = df['Answer'].values
        print("1", type(qs), qs.shape)
        print("2", type(aas), aas.shape)
        yield qs, aas


def main():

    vocab = Counter()
    questions = []
    answers = []
    
    for data_file_q, data_file_a in data():
        for q, a in zip(data_file_q, data_file_a):
            qt = word_tokenize(str(q).lower())
            at = word_tokenize(str(a).lower())

            vocab.update(qt)
            vocab.update(at)

            questions.append(qt)
            answers.append(at)
        print("Vocab:", len(vocab))

    print("vocab size:", len(vocab))
    print("questions:", len(questions))
    print("answers:", len(answers))

    with open('data/questions.pkl', 'wb') as f:
        pickle.dump(questions, f)

    with open('data/answers.pkl', 'wb') as f:
        pickle.dump(answers, f)

    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    fq = codecs.open('data/questions.txt', 'w', 'utf-8')
    fa = codecs.open('data/answers.txt', 'w', 'utf-8')
    for q, a in zip(questions, answers):
        fq.write("{}\n".format(' '.join(q)))
        fa.write("{}\n".format(' '.join(a)))
    fq.close()
    fa.close()


if __name__ == '__main__':
    main()

        
