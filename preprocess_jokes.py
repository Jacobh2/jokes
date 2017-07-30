import pandas
import csv
import string
import codecs

data = pandas.read_csv('jokes.csv')

jokes = []

print("Have", len(data), "jokes")

punc = ['.','?','!']
num_bad = 0

def replacer(string, pairs):
    for r, s in pairs:
        string = string.replace(r, s)
    return string.strip()

def process_joke(joke):
    global num_bad
    parts = joke
    if len(parts) == 2:
        # The first part
        first = parts[0]
        first = replacer(first, [
            ('...', ''), ('..', ''), ('!!!', '!'),
            ('!!', '!'), ('???', '?'), ('??', '?'),
            ('-', ''), ('_', '')
        ])
        second = parts[1]

        # The punchline needs to be short a quick
        # so make sure it only has at most one period
        second = replacer(second, [
            ('...', ''), ('..', ''), ('"', ''), ('!!!', '!'),
            ('!!', '!'), ('???', '?'), ('??', '?'), ('``', '"'),
            ("''", '"'), ('-', ''), ('_', ''), (':)', '')
        ])
        nums = list(map(lambda p: second.count(p) > 2, punc))

        # The punchline also needs to be shorted than the question
        length_diff_ok = len(second) <= 2.5*len(first)

        # The first part cannot contain too many " and no :
        first_ok = True #'"' not in first and ':' not in first

        second_ok = (second.count('"')%2 == 0) # and ':' not in second

        if '(' in second:
            second_ok = second_ok and ')' in second
        elif ')' in second:
            second_ok = second_ok and '(' in second

        if not any(nums) and length_diff_ok and first_ok and second_ok:
            return [first, second]
        num_bad += 1
        print("BAD:", num_bad)
        print("Q:", first)
        print("A:", second)
    return None

#Question,Answer
qa = data['Question'].values
aa = data['Answer'].values
for i, joke in enumerate(zip(qa, aa)):
    pj = process_joke(joke)
    if pj:
        jokes.append(pj)
    i += 1

print("Now have", len(jokes), "jokes")

#codecs.open('data/questions.txt', 'w', 'utf-8')
with codecs.open('jokes/processed2.csv', 'w', 'utf-8') as f:
    csv_file = csv.writer(f)
    csv_file.writerow(['ID','Question','Answer'])
    for i, joke in enumerate(jokes):
        row = [i]
        row.extend(joke)
        csv_file.writerow(row)
