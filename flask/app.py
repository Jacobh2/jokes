import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, render_template, request, redirect
from flask_sqlalchemy import SQLAlchemy

import runner
from reader import load_word_to_id


session, model, word_to_id, id_to_word = runner.load()


def generate_answer(input):
    return runner.answer(session, model, input, word_to_id, id_to_word)

# webapp
app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////tmp/qa.db'
db = SQLAlchemy(app)


class QA(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    question = db.Column(db.String(50), nullable=False)
    answer = db.Column(db.String(50), nullable=False)
    vote = db.Column(db.Integer, nullable=False)
    ip = db.Column(db.String(50), nullable=False)

    def __repr__(self):
        return '<IP %r>' % self.ip

# Make sure the db is created
db.create_all()

@app.route('/vote', methods=['POST'])
def vote():
    try:
        is_up = 'up' in request.form
        qa_id = request.form.get('qa_id', None)

        if qa_id is None:
            return redirect("/", code=302)

        # Load object
        qa = QA.query.get(int(qa_id))
        qa.vote = 1 if is_up else -1
        db.session.commit()
    except Exception:
        # TODO: Log to db
        pass

    return redirect("/", code=302)

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

        # Create a QA object that can be saved
        qa = QA(question=question, answer=ans, vote=0, ip=request.remote_addr)
        db.session.add(qa)
        db.session.commit()
        qa_id = qa.id

        return render_template('index.html', answer=ans, question=question, qa_id=qa_id)


if __name__ == '__main__':
    app.run()
