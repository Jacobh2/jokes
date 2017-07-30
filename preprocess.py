import pandas
import csv
import string

data = pandas.read_csv('shortjokes.csv')

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
    parts = joke.strip().split('?', 1)
    if len(parts) == 2:
        # The first part
        first = parts[0]
        
        first = replacer(first, [
            ('...', ''), ('..', ''), ('!!!', '!'),
            ('!!', '!'), ('???', '?'), ('??', '?'),
            ('-', ''), ('_', '')
        ])

        first = "{}?".format(first)
        second = parts[1]

        # The punchline needs to be short a quick
        # so make sure it only has at most one period
        second = replacer(second, [
            ('...', ''), ('..', ''), ('"', ''), ('!!!', '!'),
            ('!!', '!'), ('???', '?'), ('??', '?'), ('``', '"'),
            ("''", '"'), ('-', ''), ('_', '')
        ])
        nums = list(map(lambda p: second.count(p) > 2, punc))

        # The punchline also needs to be shorted than the question
        first_longer = len(first) >= len(second)

        # The first part cannot contain too many " and no :
        first_ok = '"' not in first and ':' not in first

        second_ok = ':' not in second and (second.count('"')%2 == 0)

        if '(' in second:
            second_ok = second_ok and ')' in second
        elif ')' in second:
            second_ok = second_ok and '(' in second

        if not any(nums) and first_longer and first_ok and second_ok:
            return [first, second]
        #num_bad += 1
        #print("BAD:", num_bad)
        #print("Q:", first)
        #print("A:", second)
    return None


for i, joke in enumerate(data['Joke']):
    pj = process_joke(joke)
    if pj:
        if '...' in pj[1]:
            print("i:", i, "pj:", pj)
            raise ValueError()
        jokes.append(pj)
    i += 1

print("Now have", len(jokes), "jokes")


with open('jokes/processed.csv', 'w', newline='') as f:
    csv_file = csv.writer(f)
    csv_file.writerow(['ID','Question','Answer'])
    for i, joke in enumerate(jokes):
        row = [i]
        row.extend(joke)
        csv_file.writerow(row)
