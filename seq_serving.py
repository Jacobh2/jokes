import os

import tensorflow as tf


def build_serving_inputs(bucket):
      serialized_input = tf.placeholder(tf.string, name='serialized_input')
      feature_configs = dict()
      names = []
      
      for i in range(bucket[0]):  
        name = "encoder{0}".format(i)
        names.append(name)
        shape = [None]
        dtype = tf.int64
        feature_configs[name] = tf.FixedLenFeature(shape=shape, dtype=dtype)
        
      parsed_example = tf.parse_example(serialized_input, feature_configs)

      encoder_inputs = []
      for name in names:
        # use tf.identity() to assign name
        encoder_inputs.append(tf.identity(parsed_example[name], name=name))

      return serialized_input, encoder_inputs


def export_model_to_serving(sess, export_path_base, version, serialized_tf_example, inputs, outputs):
    
    export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(version)))

    print('Exporting trained model to', export_path)

    builder = tf.saved_model.builder.SavedModelBuilder(export_path)

    # Build the signature_def_map.

    prediction_inputs = tf.saved_model.utils.build_tensor_info(serialized_tf_example)
    prediction_outputs =  tf.saved_model.utils.build_tensor_info(outputs)

    prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
            inputs={
                tf.saved_model.signature_constants.PREDICT_INPUTS: prediction_inputs
                },
            outputs={
                tf.saved_model.signature_constants.PREDICT_OUTPUTS: prediction_outputs
                },
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    tensor_infos_input = dict()
    for input_ in inputs:
        tensor_infos_input[input_.name] = tf.saved_model.utils.build_tensor_info(input_)
    
    tensor_infos_output = dict()
    for output in outputs:
        tensor_infos_output[output.name] = tf.saved_model.utils.build_tensor_info(output)


    add_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=tensor_infos_input,
            outputs=tensor_infos_output,
            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
    builder.add_meta_graph_and_variables(sess,
            [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'joke': add_signature,
            tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature},
            legacy_init_op=legacy_init_op)

    builder.save()

    print('Done exporting!')