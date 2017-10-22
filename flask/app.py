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
    changed = db.Column(db.Integer, nullable=False)

    def __repr__(self):
        return "id={}, q={}, a={}, ip={}".format(self.id, self.question, self.answer, self.ip)

    __str__ = __repr__

# Make sure the db is created
db.create_all()

@app.route('/vote', methods=['POST'])
def vote():
    try:
        # Get the data
        # How did the user vote
        is_up = 'up' in request.form
        # Id of the result
        qa_id = request.form.get('qa_id', None)
        # The maybe updated question
        question = request.form.get('question').strip().lower()
        # The maybe updated answer
        answer = request.form.get('answer').strip().lower()

        # sanity if we have an id
        if qa_id is None:
            return redirect("/", code=302)

        # Load object using the id
        qa = QA.query.get(int(qa_id))
        # Check if the question or answer has changed,
        # if so, it's directly an upvote
        q_diff = qa.question != question
        a_diff = qa.answer != answer

        # Set the question and answer
        qa.question = question
        qa.answer = answer
        
        # Set the vote and change flag
        if q_diff or a_diff:
            qa.vote += 1
            qa.changed = 1
        else:
            qa.vote += 1 if is_up else -1
            qa.changed = 0

        # Commit changes to db
        db.session.commit()
    except Exception as e:
        print("Error saving:", e)
        # TODO: Log to db

    return redirect("/", code=302)


@app.route('/<int:qa_id>', methods=['GET'])
def show_result(qa_id):
    # Load from the db
    qa = QA.query.get(int(qa_id))

    if qa is None:
        return redirect('/', code=302)

    # Serve the result
    return render_template('index.html', answer=qa.answer, question=qa.question, qa_id=qa_id)


@app.route('/', methods=['GET', 'POST'])
def main():
    # If we have no data, show the index page
    question = None

    if request.method == 'POST':
        question = request.form.get('question', None)
    elif request.method == 'GET':
        question = request.args.to_dict().get('question', None)

    if question is None:
        # We have no question, so just serve the index site
        return render_template('index.html')
    else:
        ans = generate_answer(question.lower())

        # Create a QA object that can be saved
        qa = QA(question=question.strip().lower(), answer=ans.strip().lower(), vote=0, ip=request.remote_addr, changed=0)
        db.session.add(qa)
        db.session.commit()
        qa_id = qa.id

        # Redirect to the specific URL so that the user can share the result bcs fun!
        return redirect('/{}'.format(qa_id), code=302)


if __name__ == '__main__':
    app.run()
