"""
This script is used to test how to implement
a string -> list of ids using tensorflow,
instead of using python code to be able to
embedd the decoding into  the model
"""

import tensorflow as tf


def test():
    
    # First build the model

    # An input to take the entire sentence
    input_sentence = tf.placeholder(tf.string, name='user_input')

    input_sentence_array = tf.reshape(input_sentence, [1])

    # Split the input into tokens
    input_tokens = tf.string_split(input_sentence_array, delimiter=' ')

    mapping_strings = tf.constant(['UNK', 'a','b','c','d','e'])
    table = tf.contrib.lookup.index_table_from_tensor(mapping=mapping_strings, num_oov_buckets=0, default_value=0)
    
    # map the tokens into IDs
    input_ids = table.lookup(input_tokens.values)


    # Now test the model:
    with tf.Session() as session:
        text_input = 'a c d e b a hejsan vad händer idag? a'
        tf.tables_initializer().run(session=session)

        result = session.run(input_ids, feed_dict={input_sentence: text_input})

        print("Result:", result)

if __name__ == '__main__':
    test()