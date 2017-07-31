#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Based in mnist_saved_model.py from Tensorflow serving example/tutorial
but a much simpler version to check if I've understood which parts are
needed and how they need to be formatted to save a model into the servings
format!
"""

import os

import tensorflow as tf


def export_model_to_serving(sess, export_path_base, serialized_tf_example, x, y, z):

    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes("1"))

    print ('Exporting trained model to', export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
    """
    # Prediction API constants.

    # Predict inputs.
    PREDICT_INPUTS = "inputs"

    # Prediction method name used in a SignatureDef.
    PREDICT_METHOD_NAME = "tensorflow/serving/predict"

    # Predict outputs.
    PREDICT_OUTPUTS = "outputs"
    """

    prediction_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    prediction_outputs =  tf.saved_model.utils.build_tensor_info(z)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.PREDICT_INPUTS: prediction_inputs
                },
            outputs={
                tf.saved_model.signature_constants.PREDICT_OUTPUTS: prediction_outputs
                },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    tensor_info_x = tf.saved_model.utils.build_tensor_info(x)
    tensor_info_y = tf.saved_model.utils.build_tensor_info(y)
    tensor_info_z = tf.saved_model.utils.build_tensor_info(z)

    add_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs={'input_x': tensor_info_x, 'input_y': tensor_info_y},
            outputs={'output': tensor_info_z},
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'add_x_and_y': add_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')


def main(_):

    with tf.Session() as session:
        # Build a model
        serialized_input = tf.placeholder(tf.string, name='serialized_input')

        feature_configs = {
            'x': tf.FixedLenFeature(shape=[], dtype=tf.float32),
            'y': tf.FixedLenFeature(shape=[], dtype=tf.float32)
            }

        parsed_example = tf.parse_example(serialized_input, feature_configs)

        # use tf.identity() to assign name
        input_x_tensor = tf.identity(parsed_example['x'], name='x')
        input_y_tensor = tf.identity(parsed_example['y'], name='y')

        print("input_x_tensor:", input_x_tensor, type(input_x_tensor))

        # target
        z = input_x_tensor + input_y_tensor

        session.run(tf.global_variables_initializer())

        print("Z:", session.run(z, feed_dict={input_x_tensor: [10.3], input_y_tensor: [22.0]}))

        export_path_base = '/tmp/serving_test'
        export_model_to_serving(session, export_path_base, serialized_input, input_x_tensor, input_y_tensor, z)


if __name__ == '__main__':
    tf.app.run()

			