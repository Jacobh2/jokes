import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request

import runner
from reader import load_word_to_id


session, model, word_to_id, id_to_word = runner.load()


def generate_answer(input):
    return runner.answer(session, model, input, word_to_id, id_to_word)

# webapp
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def main():
    # If we have no data, show the index page
    question = None

    if request.method == 'POST':
        question = request.form.get('question', None)
    elif request.method == 'GET':
        question = request.args.to_dict().get('question', None)

    if question is None:
        return render_template('index.html')
    else:
        ans = generate_answer(question.lower())
        return render_template('index.html', answer=ans, question=question)


if __name__ == '__main__':
    app.run()
