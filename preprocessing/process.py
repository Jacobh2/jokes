"""
Process the three files, tokenize and create vocab
and lastly id all tokens
"""

from preprocessing.preprocesser import Preprocesser
import pandas
import os
from nltk.tokenize import word_tokenize
import pickle
from collections import Counter
import codecs
import reader
import random


def formatter_1(data):
    questions = data['Question'].values
    answers = data['Answer'].values
    return questions, answers

def answer_ok_1(answer):
    # even number of quotes
    return (answer.count('"')%2 == 0) and (answer.count(':') < 2)


def formatter_2(data):
    jokes = data['Joke']
    questions = []
    answers = []
    for joke in jokes:
        parts = joke.strip().split('?', 1)
        if len(parts) == 2:
            questions.append(parts[0])
            answers.append(parts[1])
    return questions, answers

def question_ok_2(question):
    return '"' not in question and ':' not in question

def answer_ok_2(answer):
    # even number of quotes
    return (answer.count('"')%2 == 0) and ":" not in answer


def load_data(load_dir):
    for file_name in os.listdir(load_dir):
        path = os.path.join(load_dir, file_name)
        df = pandas.read_csv(path)
        yield df['Question'].values, df['Answer'].values


def create_vocab(load_dir, save_path, dev_part):

    vocab = Counter()
    questions = []
    answers = []
    
    for data_file_q, data_file_a in load_data(load_dir):
        for q, a in zip(data_file_q, data_file_a):
            qt = word_tokenize(str(q).lower())
            at = word_tokenize(str(a).lower())

            vocab.update(qt)
            vocab.update(at)

            questions.append(qt)
            answers.append(at)

    print("vocab size:", len(vocab))
    len_questions = len(questions)
    print("questions:", len_questions)
    print("answers:", len(answers))

    # Shuffle questions and answers!
    seed = random.random()
    random.shuffle(questions, random=lambda: seed)
    random.shuffle(answers, random=lambda: seed)

    # Split into train and dev part
    split_index = int(len_questions * (1 - dev_part))

    questions_train = questions[:split_index]
    questions_dev = questions[split_index:]

    answers_train = answers[:split_index]
    answers_dev = answers[split_index:]

    def save_pkl(data, name):
        with open(os.path.join(save_path, name), 'wb') as f:
            pickle.dump(data, f)    

    save_pkl(questions, 'questions.pkl')
    save_pkl(answers, 'answers.pkl')
    save_pkl(vocab, 'vocab.pkl')

    def save_txt(d1, d2, n1, n2):
        fq = codecs.open(os.path.join(save_path, n1), 'w', 'utf-8')
        fa = codecs.open(os.path.join(save_path, n2), 'w', 'utf-8')
        for q, a in zip(d1, d2):
            fq.write("{}\n".format(' '.join(q)))
            fa.write("{}\n".format(' '.join(a)))
        fq.close()
        fa.close()

    save_txt(questions_train, answers_train, 'questions.txt', 'answers.txt')
    save_txt(questions_dev, answers_dev, 'questions_dev.txt', 'answers_dev.txt')


if __name__ == '__main__':
    # Check if all folders exist
    if not os.path.exists('raw_data'):
        raise ValueError("No folder raw_data found")

    if not os.path.exists('processed_data'):
        print("Creating folder processed_data")
        os.mkdir('processed_data')

    if not os.path.exists('processed_data/jokes_1.csv'):
        print("Creating first preprocesser")
        pr1 = Preprocesser('raw_data/jokes.csv', 'processed_data/jokes_1.csv', formatter_1, lambda q: True, answer_ok_1)
        pr1.run()

    if not os.path.exists('processed_data/jokes_2.csv'):
        print("Creating second preprocesser")
        pr2 = Preprocesser('raw_data/jokes2.csv', 'processed_data/jokes_2.csv', formatter_1, lambda q: True, answer_ok_1)
        pr2.run()

    if not os.path.exists('processed_data/jokes_3.csv'):
        print("Creating third preprocesser")
        pr3 = Preprocesser('raw_data/shortjokes.csv', 'processed_data/jokes_3.csv', formatter_2, question_ok_2, answer_ok_2, True)
        pr3.run()

    if not os.path.exists('ided_data'):
        print("Creating ided_data")
        os.mkdir('ided_data')

    if not os.path.exists('ided_data/vocab.pkl'):
        print("Creating vocab and tokenizing data")
        create_vocab('processed_data', 'ided_data', 0.01)

    if not os.path.exists('ided_data/word_to_id.pkl'):
        print("Creating ID verison of data")
        reader.read_data('ided_data', vocab_size=25000)

