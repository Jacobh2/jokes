# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Sequence-to-sequence model with an attention mechanism."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random

import numpy as np
import tensorflow as tf
import reader

class Seq2SeqModel(object):
  """Sequence-to-sequence model with attention and for multiple buckets.

  This class implements a multi-layer recurrent neural network as encoder,
  and an attention-based decoder. This is the same as the model described in
  this paper: http://arxiv.org/abs/1412.7449 - please look there for details,
  or into the seq2seq library for complete model implementation.
  This class also allows to use GRU cells in addition to LSTM cells, and
  sampled softmax to handle large output vocabulary size. A single-layer
  version of this model, but with bi-directional encoder, was presented in
    http://arxiv.org/abs/1409.0473
  and sampled softmax is described in Section 3 of the following paper.
    http://arxiv.org/abs/1412.2007
  """

  def __init__(self, encoder_input_list, config, bucket, use_lstm=False, num_samples=512,
               forward_only=False, dtype=tf.float32):
    """Create the model.

    Args:
      use_lstm: if true, we use LSTM cells instead of GRU cells.
      num_samples: number of samples for sampled softmax.
      forward_only: if set, we do not construct the backward pass in the model.
      dtype: the data type to use to store internal variables.
    """
    self.config = config
    self.bucket = bucket
    self.learning_rate = tf.Variable(float(config.learning_rate), trainable=False, dtype=dtype)
    self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * config.learning_rate_decay_factor)
    self.global_step = tf.Variable(0, trainable=False)

    # If we use sampled softmax, we need an output projection.
    output_projection = None
    softmax_loss_function = None
    # Sampled softmax only makes sense if we sample less than vocabulary size.
    # FIXME: Implement the softmax that is needed during forward pass (running, not training)

    """
    Something like this:

    if mode == "train":
      loss = tf.nn.sampled_softmax_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          ...,
          partition_strategy="div")
    elif mode == "eval":
      logits = tf.matmul(inputs, tf.transpose(weights))
      logits = tf.nn.bias_add(logits, biases)
      labels_one_hot = tf.one_hot(labels, n_classes)
      loss = tf.nn.softmax_cross_entropy_with_logits(
          labels=labels_one_hot,
          logits=logits)

    """

    if num_samples > 0 and num_samples < self.config.vocab_size:
      w_t = tf.get_variable("proj_w", [self.config.vocab_size, self.config.size], dtype=dtype)
      w = tf.transpose(w_t)
      b = tf.get_variable("proj_b", [self.config.vocab_size], dtype=dtype)
      output_projection = (w, b)

      def sampled_loss(labels, logits):
        labels = tf.reshape(labels, [-1, 1])
        # We need to compute the sampled_softmax_loss using 32bit floats to
        # avoid numerical instabilities.
        local_w_t = tf.cast(w_t, tf.float32)
        local_b = tf.cast(b, tf.float32)
        local_inputs = tf.cast(logits, tf.float32)
        return tf.cast(
            tf.nn.sampled_softmax_loss(
                weights=local_w_t,
                biases=local_b,
                labels=labels,
                inputs=local_inputs,
                num_sampled=num_samples,
                num_classes=self.config.vocab_size),
            dtype)
      softmax_loss_function = sampled_loss

    """
    FIXME: Update to use one of GRU or LSTM, maybe even
    the new cell that you and David talked about at the start of 
    the summer? e.g. simplify this part!

    Also, simplify the "multi layered", bcs will always have num_layers > 1
    """

    # Create the internal multi-layer cell for our RNN.
    def single_cell():
      return tf.contrib.rnn.GRUCell(self.config.size)
      #return tf.contrib.rnn.GRUCell(size, reuse=tf.get_variable_scope().reuse)
    if use_lstm:
      def single_cell():
        return tf.contrib.rnn.BasicLSTMCell(self.config.size)
    cell = single_cell()
    if self.config.num_layers > 1:
      cell = tf.contrib.rnn.MultiRNNCell([single_cell() for _ in list(range(self.config.num_layers))])

    # The seq2seq function: we use embedding for the input and attention.
    def seq2seq_f(encoder_inputs, decoder_inputs, do_decode):
      return tf.contrib.legacy_seq2seq.embedding_attention_seq2seq(
          encoder_inputs,
          decoder_inputs,
          cell,
          num_encoder_symbols=self.config.vocab_size,
          num_decoder_symbols=self.config.vocab_size,
          embedding_size=self.config.size,
          output_projection=output_projection,
          feed_previous=do_decode,
          dtype=dtype)


    # Feeds for inputs.
    #self.encoder_inputs = []
    self.encoder_inputs = encoder_input_list
    self.decoder_inputs = []
    self.target_weights = []
    #for i in range(bucket[0]):  
    #  self.encoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
    #                                            name="encoder{0}".format(i)))
    for i in range(bucket[1] + 1):
      self.decoder_inputs.append(tf.placeholder(tf.int32, shape=[None],
                                                name="decoder{0}".format(i)))
      self.target_weights.append(tf.placeholder(dtype, shape=[None],
                                                name="weight{0}".format(i)))

    # Our targets are decoder inputs shifted by one.
    targets = [self.decoder_inputs[i + 1]
               for i in range(len(self.decoder_inputs) - 1)]

    # Training outputs and losses.
    if forward_only:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, [bucket], lambda x, y: seq2seq_f(x, y, True),
          softmax_loss_function=softmax_loss_function)
      # If we use output projection, we need to project outputs for decoding.
      if output_projection is not None:
        self.outputs[0] = [tf.matmul(output, output_projection[0]) + output_projection[1] for output in self.outputs[0]]
    else:
      self.outputs, self.losses = tf.contrib.legacy_seq2seq.model_with_buckets(
          self.encoder_inputs, self.decoder_inputs, targets,
          self.target_weights, [bucket],
          lambda x, y: seq2seq_f(x, y, False),
          softmax_loss_function=softmax_loss_function)

    # Gradients and SGD update operation for training the model.
    params = tf.trainable_variables()
    if not forward_only:
      self.gradient_norms = []
      self.updates = []
      opt = tf.train.GradientDescentOptimizer(self.learning_rate)
      gradients = tf.gradients(self.losses[0], params)
      clipped_gradients, norm = tf.clip_by_global_norm(gradients, self.config.max_gradient_norm)
      self.gradient_norms.append(norm)
      self.updates.append(opt.apply_gradients(
          zip(clipped_gradients, params), global_step=self.global_step))

    self.saver = tf.train.Saver(tf.global_variables())

  def step(self, session, encoder_inputs, decoder_inputs, target_weights, forward_only):
    """Run a step of the model feeding the given inputs.

    Args:
      session: tensorflow session to use.
      encoder_inputs: list of numpy int vectors to feed as encoder inputs.
      decoder_inputs: list of numpy int vectors to feed as decoder inputs.
      target_weights: list of numpy float vectors to feed as target weights.
      forward_only: whether to do the backward step or only forward.

    Returns:
      A triple consisting of gradient norm (or None if we did not do backward),
      average perplexity, and the outputs.

    Raises:
      ValueError: if length of encoder_inputs, decoder_inputs, or
        target_weights disagrees with bucket size.
    """

    """
    FIXME: Remove everything that has to do with buckets, which will
    simplify the output tensor. Also, find out which output tensor
    is best to return for the serving. Maybe even do some post-processing
    where the ID's are converted back to words? (Also add to the model
    where it can accept a list of tokens instead of ids)
    """
    # Check if the sizes match.
    encoder_size, decoder_size = self.bucket
    if len(encoder_inputs) != encoder_size:
      raise ValueError("Encoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(encoder_inputs), encoder_size))
    if len(decoder_inputs) != decoder_size:
      raise ValueError("Decoder length must be equal to the one in bucket,"
                       " %d != %d." % (len(decoder_inputs), decoder_size))
    if len(target_weights) != decoder_size:
      raise ValueError("Weights length must be equal to the one in bucket,"
                       " %d != %d." % (len(target_weights), decoder_size))

    # Input feed: encoder inputs, decoder inputs, target_weights, as provided.
    input_feed = {}
    for l in range(encoder_size):
      input_feed[self.encoder_inputs[l].name] = encoder_inputs[l]
    for l in range(decoder_size):
      input_feed[self.decoder_inputs[l].name] = decoder_inputs[l]
      input_feed[self.target_weights[l].name] = target_weights[l]

    # Since our targets are decoder inputs shifted by one, we need one more.
    last_target = self.decoder_inputs[decoder_size].name
    input_feed[last_target] = np.zeros([self.config.batch_size], dtype=np.int32)

    # Output feed: depends on whether we do a backward step or not.
    if not forward_only:
      output_feed = [self.updates[0],  # Update Op that does SGD.
                     self.gradient_norms[0],  # Gradient norm.
                     self.losses[0]]  # Loss for this batch.
    else:
      output_feed = [self.losses[0]]  # Loss for this batch.
      for l in range(decoder_size):  # Output logits.
        output_feed.append(self.outputs[0][l])

    outputs = session.run(output_feed, input_feed)
    if not forward_only:
      return outputs[1], outputs[2], None  # Gradient norm, loss, no outputs.
    else:
      return None, outputs[0], outputs[1:]  # No gradient norm, loss, outputs.

  def get_batch(self, data):
    """Get a random batch of data from the specified bucket, prepare for step.

    To feed data in step(..) it must be a list of batch-major vectors, while
    data here contains single length-major cases. So the main logic of this
    function is to re-index data cases to be in the proper format for feeding.

    Args:
      data: a tuple of size len(self.buckets) in which each element contains
        lists of pairs of input and output data that we use to create a batch.

    Returns:
      The triple (encoder_inputs, decoder_inputs, target_weights) for
      the constructed batch that has the proper format to call step(...) later.
    """
    encoder_size, decoder_size = self.bucket
    encoder_inputs, decoder_inputs = [], []

    # Get a random batch of encoder and decoder inputs from data,
    # pad them if needed, reverse encoder inputs and add GO to decoder.
    for _ in range(self.config.batch_size):
      encoder_input, decoder_input = random.choice(data[0])

      # Encoder inputs are padded and then reversed.
      encoder_pad = [reader.PAD_ID] * (encoder_size - len(encoder_input))
      encoder_inputs.append(list(reversed(encoder_input + encoder_pad)))

      # Decoder inputs get an extra "GO" symbol, and are padded then.
      decoder_pad_size = decoder_size - len(decoder_input) - 1
      decoder_inputs.append([reader.GO_ID] + decoder_input +
                            [reader.PAD_ID] * decoder_pad_size)

    # Now we create batch-major vectors from the data selected above.
    batch_encoder_inputs, batch_decoder_inputs, batch_weights = [], [], []

    # Batch encoder inputs are just re-indexed encoder_inputs.
    for length_idx in range(encoder_size):
      batch_encoder_inputs.append(
          np.array([encoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.config.batch_size)], dtype=np.int32))

    # Batch decoder inputs are re-indexed decoder_inputs, we create weights.
    for length_idx in range(decoder_size):
      batch_decoder_inputs.append(
          np.array([decoder_inputs[batch_idx][length_idx]
                    for batch_idx in range(self.config.batch_size)], dtype=np.int32))

      # Create target_weights to be 0 for targets that are padding.
      batch_weight = np.ones(self.config.batch_size, dtype=np.float32)
      for batch_idx in range(self.config.batch_size):
        # We set weight to 0 if the corresponding target is a PAD symbol.
        # The corresponding target is decoder_input shifted by 1 forward.
        if length_idx < decoder_size - 1:
          target = decoder_inputs[batch_idx][length_idx + 1]
        if length_idx == decoder_size - 1 or target == reader.PAD_ID:
          batch_weight[batch_idx] = 0.0
      batch_weights.append(batch_weight)
    return batch_encoder_inputs, batch_decoder_inputs, batch_weights