"""
How to process the jokes files
"""

import pandas
import csv
import string
import codecs


class Preprocesser(object):
    
    data = None
    jokes = None
    bad_lines = 0
    punc_chars = ['.', '?', '!']
    replacers = [
            ('...', ''), ('..', ''), ('"', ''), ('!!!', '!'),
            ('!!', '!'), ('???', '?'), ('??', '?'), ('``', '"'),
            ("''", '"'), ('-', ''), ('_', ''), (':)', ''), ('Q: ', ''),
            ('A: ', '')
        ]
    
    def __init__(self, file_path, save_path, formatter, question_ok_fn, answer_ok_fn, question_append=False):
        self.file_path = file_path
        self.save_path = save_path
        self.formatter = formatter
        self.question_ok_fn = question_ok_fn
        self.answer_ok_fn = answer_ok_fn
        self.question_append = question_append

        self.jokes = []
        self.data = None
        self.bad_lines = 0

    def load(self):
        self.data = pandas.read_csv(self.file_path)

    def replacer(self, string, pairs):
        for repl, swith in pairs:
            string = string.replace(repl, swith)
        return string.strip()

    def parser(self, joke_parts):
        if len(joke_parts) != 2:
            self.bad_lines += 1
            return

        # Get the two parts
        question, answer = joke_parts

        # Fix the question
        question = self.replacer(question, self.replacers)

        if self.question_append:
            question = "{}?".format(question)

        que_length_ok = len(question) > 3
        question_ok = que_length_ok and self.question_ok_fn(question)

        # Fix the answer
        answer = self.replacer(answer, self.replacers)
        
        # Count the number of punctuations in the answer part.
        # If there are more than 2 of one specific punctuation,
        # then its most prob not a good que/ans joke
        punct_num = list(map(lambda p: answer.count(p) > 2, self.punc_chars))
        
        # The punchline cannot be too long,
        # compared to the question
        length_diff_ok = len(answer) <= 2.5*len(question)

        # Check so there actually exists a joke
        ans_length_ok = len(answer) > 3

        # Check any specific
        answer_ok = ans_length_ok and self.answer_ok_fn(answer)

        # Check the number of start/end parentheses
        if '(' in answer:
            answer_ok = answer_ok and ')' in answer
        elif ')' in answer:
            answer_ok = answer_ok and '(' in answer

        # Check all conditions
        if not any(punct_num) and length_diff_ok and answer_ok and question_ok:
            self.jokes.append((question, answer))
        else:
            self.bad_lines += 1

    def run(self):
        if self.data is None:
            self.load()

        #Question,Answer
        formatted_data = self.formatter(self.data)
        list(map(self.parser, zip(*formatted_data)))

        num_jokes = len(self.jokes)
        if num_jokes == 0:
            print("No jokes :(")
        else:
            print("Have", num_jokes, "jokes")

        print("Num bad lines:", self.bad_lines)

        self.save()
    
    def save(self):
        with codecs.open(self.save_path, 'w', 'utf-8') as f:
            csv_file = csv.writer(f)
            csv_file.writerow(['ID','Question','Answer'])
            for i, joke in enumerate(self.jokes):
                row = [i]
                row.extend(joke)
                csv_file.writerow(row)
