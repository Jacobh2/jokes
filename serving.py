import os

import tensorflow as tf

def export_model_to_serving(sess, export_path_base, model_version, serialized_tf_example, x, y, z):
    
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(model_version))

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.
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