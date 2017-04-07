import math
import os
import random
import sys
import time
import logging

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import data_utils
import seq2seq_model
from config import *

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
	data_set = [[] for _ in BUCKETS]
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
				target_ids.append(data_utils.EOS_ID)
				for bucket_id, (source_size, target_size) in enumerate(BUCKETS):
					if len(source_ids) < source_size and len(target_ids) < target_size:
						data_set[bucket_id].append([source_ids, target_ids])
						break
				source, target = source_file.readline(), target_file.readline()
	return data_set

def create_model(session, conf={}):

  	"""
  		Create translation model and initialize or load parameters in session.
		session = tensorflow session
		config = set of parameters that define the model
	"""
	if len(conf) == 0:
		conf ={}
	conf.setdefault('hidden_size', 256)
	conf.setdefault('num_layers', 1)
	conf.setdefault('batch_size', 2)
	conf.setdefault('learning_rate', 0.005)
	conf.setdefault('learning_rate_decay', 0.99)
	conf.setdefault('max_gradient_norm', 5.0)
	conf.setdefault('test', False)
	conf.setdefault('dtype', tf.float32)
	conf.setdefault('use_lstm', True)

	dtype = conf['dtype']
	
	print "Creating", conf['hidden_size'], " of", conf['num_layers'], "layers"

	model = seq2seq_model.Seq2SeqModel(
		V,
		V,
		BUCKETS,
		conf['hidden_size'],
		conf['num_layers'],
		conf['max_gradient_norm'],
		conf['batch_size'],
		conf['learning_rate'],
		conf['learning_rate_decay'],
		use_lstm=conf['use_lstm'],
		forward_only=conf['test'],
		embed_size=EMBED_SIZE,
		dtype=dtype)

	ckpt = tf.train.get_checkpoint_state(MODELS_DIR)
	if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
		print "Reading model parameters from %s" % ckpt.model_checkpoint_path
		model.saver.restore(session, ckpt.model_checkpoint_path)
	else:
		print "Created model with fresh parameters."
		session.run(tf.global_variables_initializer())

	return model



def prepare():

	from_train, to_train, from_dev, to_dev, _ = data_utils.custom_prepare_data(
		DATA_DIR,
		QUES_TRAIN_FILE,
		ANS_TRAIN_FILE,
		QUES_DEV_FILE,
		ANS_DEV_FILE,
		V,
		VOCAB_PATH)

	dev_set = read_data(from_dev, to_dev)
	train_set = read_data(from_train, to_train, MAX_TRAIN_SIZE)
	train_bucket_sizes = [len(train_set[b]) for b in xrange(len(BUCKETS))]
	train_total_size = float(sum(train_bucket_sizes))

	# A bucket scale is a list of increasing numbers from 0 to 1 that we'll use
	# to select a bucket. Length of [scale[i], scale[i+1]] is proportional to
	# the size if i-th training bucket, as used later.
	train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
							for i in xrange(len(train_bucket_sizes))]

	return [train_set, dev_set, train_bucket_sizes, train_total_size, train_buckets_scale]

def train(session, model, train_data):

	"""
		train = [ques_train, ans_train]
		dev = [ques_dev, ans_dev]

		Returns the trained model
	"""

	train_set, dev_set, train_bucket_sizes, train_total_size, train_buckets_scale = train_data
	vocab_dict, reverse_vocab = data_utils.initialize_vocabulary(VOCAB_PATH)
	
	step_time, loss = 0.0, 0.0
	current_step = 0
	previous_losses = []

	while True:
		# Choose a bucket according to data distribution. We pick a random number
		# in [0, 1] and use the corresponding interval in train_buckets_scale.
		random_number_01 = np.random.random_sample()
		bucket_id = min([i for i in xrange(len(train_buckets_scale))
		           if train_buckets_scale[i] > random_number_01])

		# Get a batch and make a step.
		start_time = time.time()
		encoder_inputs, decoder_inputs, target_weights = model.get_batch(
		train_set, bucket_id)
		_, step_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
		                       target_weights, bucket_id, False)
		step_time += (time.time() - start_time) / STEPS_PER_CHECKPOINT
		loss += step_loss / STEPS_PER_CHECKPOINT
		current_step += 1

		if current_step % STEPS_PER_CHECKPOINT == 0:
			print "Testing the model"
			perplexity = math.exp(float(loss)) if loss < 300 else float("inf")
			print ("global step %d learning rate %.4f step-time %.2f perplexity "
					"%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
							step_time, perplexity))
			# Decrease learning rate if no improvement was seen over last 3 times.
			if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
				session.run(model.learning_rate_decay_op)
			previous_losses.append(loss)
			# Save checkpoint and zero timer and loss.
			checkpoint_path = os.path.join(MODELS_DIR, "chatbot.ckpt")
			model.saver.save(session, checkpoint_path, global_step=model.global_step)
			step_time, loss = 0.0, 0.0
			# Run evals on development set and print their perplexity.
			for bucket_id in xrange(len(BUCKETS)):
				if len(dev_set[bucket_id]) == 0:
					print("  eval: empty bucket %d" % (bucket_id))
					continue
				encoder_inputs, decoder_inputs, target_weights = model.get_batch(
				  dev_set, bucket_id)
				_, eval_loss, _ = model.step(session, encoder_inputs, decoder_inputs,
				                           target_weights, bucket_id, True)
				eval_ppx = math.exp(float(eval_loss)) if eval_loss < 300 else float(
				  "inf")
				print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
			sys.stdout.flush()

	return model

def softmax(scores):

	scores -= max(scores)
	scores = np.exp(scores)
	scores /= np.sum(scores)
	return scores

def test(session, model, sentence):

	"""
		Make sure that the model is created again with test = True
		make sure that model.batch_size = 1
	"""
	model.batch_size = 1
	vocab_dict, reverse_vocab = data_utils.initialize_vocabulary(VOCAB_PATH)
	print "Understanding question"
	# Get token-ids for the input sentence.
	token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), vocab_dict)
	
	print "Tokenized:", token_ids

	# Which bucket does it belong to?
	bucket_id = len(BUCKETS) - 1
	for i, bucket in enumerate(BUCKETS):
		if bucket[0] >= len(token_ids):
			bucket_id = i
			break
		else:
			logging.warning("Sentence truncated: %s", sentence)

	# Get a 1-element batch to feed the sentence to the model.
	encoder_inputs, decoder_inputs, target_weights = model.get_batch(
	{bucket_id: [(token_ids, [])]}, bucket_id)

	# Get output logits for the sentence.
	_, _, output_logits = model.step(session, encoder_inputs, decoder_inputs,
	                           target_weights, bucket_id, True)
	# This is a greedy decoder - outputs are just argmaxes of output_logits.
	# outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
	# print len(output_logits)
	# print output_logits[0][0,:].shape
	outputs = [int(np.random.choice(logit.shape[1], 1, p=softmax(logit[0,:]))) for logit in output_logits]

	# If there is an EOS symbol in outputs, cut them at that point.
	if data_utils.EOS_ID in outputs:
		outputs = outputs[:outputs.index(data_utils.EOS_ID)]

	return " ".join([tf.compat.as_str(reverse_vocab[output]) for output in outputs])

if __name__ == "__main__":
	data = prepare()
	with tf.Session() as sess:
		model = create_model(sess, conf={'test':True})
		# model = train(sess, model, data)
		while True:
			sent = raw_input("You:")
			print test(sess, model, sent)