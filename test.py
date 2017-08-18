import seq_serving
import tensorflow as tf


input_config = (2, 2)

serialized_input, encoder_inputs = seq_serving.build_serving_inputs(input_config[0], verbose=True)

# Create using the old code
encoder_inputs_orig = []
for i in range(input_config[0]):  # Last bucket is the biggest one.
    encoder_inputs_orig.append(tf.placeholder(tf.int32, shape=[None], name="encoder{0}".format(i)))

print("A:", encoder_inputs)

print("-"*32)

print("B:", encoder_inputs_orig)