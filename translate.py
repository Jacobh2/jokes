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

"""Binary for training translation models and decoding from them.

Running this program without --decode will download the WMT corpus into
the directory specified as --data_dir and tokenize it in a very basic way,
and then start training a model saving checkpoints to --train_dir.

Running with --decode starts an interactive loop so you can see how
the current checkpoint translates English sentences into French.

See the following papers for more information on neural translation models.
 * http://arxiv.org/abs/1409.3215
 * http://arxiv.org/abs/1409.0473
 * http://arxiv.org/abs/1412.2007
"""
import math
import os
import random
import sys
import time
import logging

import numpy as np
import tensorflow as tf

import reader
import seq2seq_model
import configuration



tf.app.flags.DEFINE_integer("version", 1, "Version of the network")
tf.app.flags.DEFINE_string("export_dir", "/tmp/seqserving", "Export directory")
tf.app.flags.DEFINE_string("data_dir", "./data_small", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "./save", "Training directory.")
tf.app.flags.DEFINE_integer("max_train_data_size", 0, "Limit on the size of training data (0: no limit).")
tf.app.flags.DEFINE_boolean("decode", False, "Set to True for interactive decoding.")
tf.app.flags.DEFINE_boolean("use_fp16", False, "Train using fp16 instead of fp32.")
tf.app.flags.DEFINE_integer("steps_per_checkpoint", 200,
                            "How many training steps to do per checkpoint.")

FLAGS = tf.app.flags.FLAGS

# We use a number of buckets and pad to the closest one for efficiency.
# See seq2seq_model.Seq2SeqModel for details of how they work.
_buckets = [(50, 50)]


def read_data(source_path, target_path, max_size=None):
  """Read data from source and target files and put into buckets.

  Args:
    source_path: path to the files with token-ids for the source language.
    target_path: path to the file with token-ids for the target language;
      it must be aligned with the source file: n-th line contains the desired
      output for n-th line from the source_path.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      counter = 0
      while source and target and (not max_size or counter < max_size):
        counter += 1
        if counter % 100000 == 0:
          print("  reading data line %d" % counter)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(reader.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only, config):
  """Create translation model and initialize or load parameters in session."""
  dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
  model = seq2seq_model.Seq2SeqModel(
      config.vocab_size,
      _buckets,
      config.size,
      config.num_layers,
      config.max_gradient_norm,
      config.batch_size,
      config.learning_rate,
      config.learning_rate_decay_factor,
      forward_only=forward_only,
      dtype=dtype)
  ckpt = tf.train.get_checkpoint_state(FLAGS.train_dir)
  if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.global_variables_initializer())
  return model


def train():
  """Train a en->fr translation model using WMT data."""
  from_train = os.path.join(FLAGS.data_dir, 'questions.txt.ids')
  to_train = os.path.join(FLAGS.data_dir, 'answers.txt.ids')
  from_dev = os.path.join(FLAGS.data_dir, 'questions_dev.txt.ids')
  to_dev = os.path.join(FLAGS.data_dir, 'answers_dev.txt.ids')

  # Create config
  config = configuration.Config()

  with tf.Session() as sess:
    # Create model.
    print("Creating %d layers of %d units." % (config.num_layers, config.size))
    model = create_model(sess, False, config)

    # Read data into buckets and compute their sizes.
    print ("Reading development and training data (limit: %d)."
           % FLAGS.max_train_data_size)
    dev_set = read_data(from_dev, to_dev)
    train_set = read_data(from_train, to_train, FLAGS.max_train_data_size)
    train_bucket_sizes = [len(train_set[b]) for b in range(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    # A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
    # to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
    # the size if i-th training bucket, as used later.
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in range(len(train_bucket_sizes))]

    # This is the training loop.
    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []

    max_epochs = config.epochs
    total_train_steps = math.ceil(train_total_size/config.batch_size)
    checkpoint_path = os.path.join(FLAGS.train_dir, "translate.ckpt")

    print("Train total size:", train_total_size)
    print("train_buckets_scale", train_buckets_scale)
    for epoch in range(max_epochs):
      for train_step in range(total_train_steps):
        # Choose a bucket according to data distribution. We pick a random number
        # in [0, 1] and use the corresponding interval in train_buckets_scale.
        random_number_01 = np.random.random_sample()
        bucket_id = min([i for i in range(len(train_buckets_scale))
                        if train_buckets_scale[i] > random_number_01])

        # Get a batch and make a step.
        start_time = time.time()
        encoder_inputs, decoder_inputs, target_weights = model.get_batch(
            train_set, bucket_id)
        _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                    target_weights, bucket_id, False)
        step_time += (time.time() - start_time) / FLAGS.steps_per_checkpoint
        loss += step_loss / FLAGS.steps_per_checkpoint
        current_step += 1

        # Once in a while, we save checkpoint, print statistics, and run evals.
        if current_step % FLAGS.steps_per_checkpoint == 0:
          # Print statistics for the previous epoch.
          perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
          print ("global step %d learning rate %.4f step-time %.2f perplexity "
                "%.2f loss %.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                          step_time, perplexity, loss))
          # Decrease learning rate if no improvement was seen over last 3 times.
          if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
            sess.run(model.learning_rate_decay_op)
          previous_losses.append(loss)
          # Save checkpoint and zero timer and loss.
          model.saver.save(sess, checkpoint_path, global_step=model.global_step)
          step_time, loss = 0.0, 0.0
          # Run evals on development set and print their perplexity.
          for bucket_id in range(len(_buckets)):
            if len(dev_set[bucket_id]) == 0:
              print("  eval: empty bucket %d" % (bucket_id))
              continue
            encoder_inputs, decoder_inputs, target_weights = model.get_batch(
                dev_set, bucket_id)
            _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                        target_weights, bucket_id, True)
            eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
                "inf")
            print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
          sys.stdout.flush()
      # End of epoch, save checkpoint
      print("Epoch", epoch+1, "done, saving checkpoint")
      model.saver.save(sess, checkpoint_path, global_step=model.global_step)
    print("Training done! Saving checkpint")
    model.saver.save(sess, checkpoint_path, global_step=model.global_step)



def decode():
  config = configuration.Config()
  with tf.Session() as sess:
    # Create model and load parameters.
    model = create_model(sess, True, config)
    model.batch_size = 1  # We decode one sentence at a time.

    # Load vocabularies.
    word_to_id = reader.load_word_to_id(FLAGS.data_dir)
    id_to_word = dict([(v, k) for k, v in word_to_id.items()])

    # Decode from standard input.
    sentence = input("> ").lower()
    while sentence != 'q':
      # Get token-ids for the input sentence.
      token_ids = reader.convert_to_id([sentence], word_to_id)[0].split(' ')
      # Which bucket does it belong to?
      bucket_id = len(_buckets) - 1
      for i, bucket in enumerate(_buckets):
        if bucket[0] >= len(token_ids):
          bucket_id = i
          break
      else:
        logging.warning("Sentence truncated: %s", sentence)

      # Get a 1-element batch to feed the sentence to the model.
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          {bucket_id: [(token_ids, [])]}, bucket_id)
      # Get output logits for the sentence.
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
      # This is a greedy decoder - outputs are just argmaxes of output_logits.
      outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]


      # Custom decoder, weighted pick
      """
      outputs = []
      for logit in output_logits:
        pick = weighted_pick(logit[0])
        max_itt = 0
        while pick in reader.START_VOCAB_ID:
          if max_itt > 5:
            print("Max try")
            break
          pick = weighted_pick(logit[0])
          max_itt += 1
        print("Pick:", pick, type(pick))
        outputs.append(pick)
      #outputs = [int(weighted_pick(logit[0:,1])) for logit in output_logits]
      """
      

      # If there is an EOS symbol in outputs, cut them at that point.
      if reader.EOS_ID in outputs:
        outputs = outputs[:outputs.index(reader.EOS_ID)]
      # Print out French sentence corresponding to outputs.
      print(" ".join([tf.compat.as_str(id_to_word[output]) for output in outputs]))
      sentence = input("> ").lower()


def self_test():
  """Test the translation model."""
  with tf.Session() as sess:
    print("Self-test for neural translation model.")
    # Create model with vocabularies of 10, 2 small buckets, 2 layers of 32.
    model = seq2seq_model.Seq2SeqModel(10, 10, [(3, 3), (6, 6)], 32, 2,
                                       5.0, 32, 0.3, 0.99, num_samples=8)
    sess.run(tf.global_variables_initializer())

    # Fake data set for both the (3, 3) and (6, 6) bucket.
    data_set = ([([1, 1], [2, 2]), ([3, 3], [4]), ([5], [6])],
                [([1, 1, 1, 1, 1], [2, 2, 2, 2, 2]), ([3, 3, 3], [5, 6])])
    for _ in range(5):  # Train the fake model for 5 steps.
      bucket_id = random.choice([0, 1])
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          data_set, bucket_id)
      model.step(sess, encoder_inputs, decoder_inputs, target_weights,
                 bucket_id, False)


def main(_):
  if FLAGS.decode:
    decode()
  else:
    train()

if __name__ == "__main__":
  tf.app.run()