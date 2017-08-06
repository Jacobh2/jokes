# Sequence to sequence deep RNN
A network based on the the seq2seq tutorial by Tensorflow (https://www.tensorflow.org/tutorials/seq2seq)

## Dataset
The dataset used is in the "Question and Answer" format, meaning
the jokes are for example:

_Why do chicken coops only have two doors? Because if they had four, they would be chicken sedans!_

where the input to en encoder is the first part of the joke, the "question" part, and
the second part of the joke, the "punchline" is the answer part for the decoder!

Will push dataset once it is correctly cleaned! :)

The raw data can be found over at https://www.kaggle.com. There were two datasets used: One CSV formatted like "ID, Question, Answer", and another with just "ID, Joke". The preprocessing step will normalize the second datafile so that both have the same format.

## Bazel + Tensorflow Serving
This repo also includes code for building the model together with Tensorflow Servings
using Bazel.

It also includes a simple servings-client that can be used to call the server running the
trained model to make yourself laugh



