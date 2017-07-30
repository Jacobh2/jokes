#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
Client for simple_test

Typical usage example:

    mnist_client.py --num_tests=100 --server=localhost:9000
"""

from __future__ import print_function

import sys

from grpc.beta import implementations
import numpy
import tensorflow as tf

from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import time

tf.app.flags.DEFINE_integer('concurrency', 1, 'maximum number of concurrent inference requests')
tf.app.flags.DEFINE_integer('num_tests', 100, 'Number of test images')
tf.app.flags.DEFINE_string('server', '', 'PredictionService host:port')
tf.app.flags.DEFINE_string('work_dir', '/tmp', 'Working directory. ')
FLAGS = tf.app.flags.FLAGS


def _create_rpc_callback(result_future):
    """Creates RPC callback function.

  Args:
    label: The correct label for the predicted example.
    result_counter: Counter for the prediction result.
  Returns:
    The callback function.
  """
    exception = result_future.exception()
    if exception:
        print(exception)
    else:
        print("Has result and no error!")
        response = result_future.result().outputs['output'].float_val
        print('Response:', response, type(response), ":", numpy.array(response))


def do_inference(hostport, work_dir, concurrency, num_tests):
    """Tests PredictionService with concurrent requests.

    Args:
      hostport: Host:port address of the PredictionService.
      work_dir: The full path of working directory for test data set.
      concurrency: Maximum number of concurrent requests.
      num_tests: Number of test images to use.

    Returns:
      The classification error rate.

    Raises:
      IOError: An error occurred processing test data set.
    """

    (host, port) = hostport.split(':')
    channel = implementations.insecure_channel(host, int(port))
    stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)

    num_x = float(input("Enter x: "))
    num_y = float(input("Enter y: "))

    request = predict_pb2.PredictRequest()
    # The name of the model
    request.model_spec.name = 'simple_test'
    # The name of the "function"
    request.model_spec.signature_name = 'add_x_and_y'
    
    # tf.contrib.util.make_tensor_proto(values, dtype=None, shape=None, verify_shape=False)
    request.inputs['input_x'].CopyFrom(tf.contrib.util.make_tensor_proto([num_x], dtype=tf.float32, shape=[]))
    #TensorProto
    request.inputs['input_y'].CopyFrom(tf.contrib.util.make_tensor_proto([num_y], dtype=tf.float32, shape=[]))

    result_future = stub.Predict.future(request, 5.0)  # 5 seconds timeout
    result_future.add_done_callback(_create_rpc_callback)

    print("SLEEP")
    time.sleep(10)
    


def main(_):
    if FLAGS.num_tests > 10000:
        print('num_tests should not be greater than 10k')
        return
    if not FLAGS.server:
        print('please specify server host:port')
        return
    do_inference(FLAGS.server, FLAGS.work_dir, FLAGS.concurrency, FLAGS.num_tests)


if __name__ == '__main__':
    tf.app.run()

			