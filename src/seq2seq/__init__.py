import tensorflow as tf
from nn import Encoder, Context, Decoder

class Seq2Seq(object):

	def __init__(self, graph, source_vocab_size, enc_size, enc_layers, enc_max_length,
				context_size, target_vocab_size, dec_size, dec_layers, dec_max_length,
				cell_type='LSTM', embed_size=None, train_embed=True):

		# Initialize all the passed variables
		self.graph = graph
		self.source_vocab_size = source_vocab_size
		self.enc_size = enc_size
		self.enc_layers = enc_layers
		self.enc_max_length = enc_max_length
		self.context_size = context_size
		self.target_vocab_size = target_vocab_size
		self.dec_size = dec_size
		self.dec_layers = dec_layers
		self.dec_max_length = dec_max_length
		self.cell_type = cell_type
		self.embed_size = embed_size

		
		with self.graph.as_default():
			# Create encoder namescope
			with tf.variable_scope('Encoder'):
				self.encoder = Encoder(graph, source_vocab_size, enc_size, enc_layers, enc_max_length,
									cell_type, embed_size, train_embed)

			# Context namescope
			with tf.variable_scope('Context'):
				self.context = Context(graph, self.encoder.enc_states, enc_size, dec_size, context_size)

			# Decoder namescope
			with tf.variable_scope('Decoder'):
				self.decoder = Decoder(graph, self.context, target_vocab_size, enc_size, dec_size,
									embed_size, dec_layers, dec_max_length, self.encoder.embedding, cell_type)

			self.loss = self.decoder.total_avg_loss

			# Either get all trainable variables here and apply the gradient or directly optimize the loss
			opt = tf.train.AdamOptimizer() # Use default hyperparams for now
			train_step = opt.minimize(self.loss)

	def train_step(self, session):

		feed_dict = None
		_, loss = session.run([self.opt_step, self.loss], feed_dict)
		return loss

	def test_step(self, test_type='beam'):
		# Specify beam or greedy decode

		def op(test_type):
			if test_type == 'beam':
				return self.decoder.beam_decode
			else:
				return self.decoder.greedy_decode
		feed_dict = None
		outputs = session.run(op(test_type), feed_dict)

if __name__ == '__main__':

	import numpy as np

	source_vocab_size = 10
	enc_size = 20
	enc_layers = 1
	enc_max_length = 20 
	context_size = 20
	target_vocab_size = source_vocab_size 
	dec_size = 20
	dec_layers = 1
	dec_max_length = 20 
	cell_type = 'LSTM'
	embed_size = 30
	train_embed = True

	graph = tf.Graph()
	with tf.Session(graph=graph) as sess:
		model = Seq2Seq(graph, source_vocab_size, enc_size, enc_layers, enc_max_length,
				context_size, target_vocab_size, dec_size, dec_layers, dec_max_length,
				cell_type, embed_size, train_embed)
		merged = tf.summary.merge_all()
		writer = tf.summary.FileWriter('../logs/',
                                      sess.graph)
		init_op = tf.global_variables_initializer()
		sess.run(init_op)	
