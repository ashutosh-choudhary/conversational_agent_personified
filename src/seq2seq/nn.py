import tensorflow as tf
import cPickle as pickle
from tensorflow.python.ops import variable_scope
import numpy as np

class Encoder(object):

	def __init__(self, graph, source_vocab_size, state_size, num_layers, max_length,
				cell_type='LSTM', embed_size=None, train_embed=True):

		self.graph = graph
		with self.graph.as_default():
			
			self.cell = _create_cell(state_size, num_layers, cell_type)

			self.max_length = max_length
			# Initialize the embedding
			self.embedding = None
			self.inp_dims = source_vocab_size
			if embed_size:
				self.inp_dims = embed_size
				if train_embed is False:
					embeddings_matrix = np.array(pickle.load(open('../../res/embed_weights.pkl')))
				else:
					embeddings_matrix = np.random.rand(source_vocab_size, embed_size)

				self.embedding = variable_scope.get_variable("embedding",
                                            [source_vocab_size, embed_size],
                                            initializer=tf.constant_initializer(embeddings_matrix),
                                            trainable=train_embed)
			
			# Create placeholders for encoder_inputs and lengths
			self.encoder_inputs = tf.placeholder(tf.int32, [None, max_length], name='enc_inputs')
			self.encoder_lengths = tf.placeholder(tf.float64, [None], name='enc_lengths')
			
			if self.embedding:
				#Create embedding lookup function for the entire batch
				self.embed_inputs = []
				for t in xrange(max_length):
					encoder_inp = self.encoder_inputs[:, t]
					self.embed_inputs.append(tf.cast(tf.nn.embedding_lookup(params=self.embedding, ids=encoder_inp), tf.float64)) 
				# Transpose the time axis so we have shape NxTxD tensor
				self.embed_inputs = tf.transpose(tf.stack(self.embed_inputs), perm=[1,0,2])
			else:
				self.embed_inputs = tf.cast(self.encoder_inputs, tf.float64)
				# Need to reshape to work with dynamic_rnn input
				self.embed_inputs = tf.reshape(self.embed_inputs, [-1, max_length, 1])

			self.enc_states, _ = tf.nn.dynamic_rnn(
							    cell=self.cell,
							    dtype=tf.float64,
							    sequence_length=self.encoder_lengths,
							    inputs=self.embed_inputs)

	def encode(self, session, enc_inputs, enc_lengths):
		
		input_feed = {self.encoder_inputs: enc_inputs, self.encoder_lengths:enc_lengths}
		results = session.run(self.enc_states, feed_dict=input_feed)
		
		return results

class Context(object):

	def __init__(self, graph, hstates, encoder_size, decoder_size, context_size):
		
		self.graph = graph
		self.context_size = context_size
		with self.graph.as_default():
			self.V = variable_scope.get_variable("context_V",
	                                            [context_size, decoder_size],
	                                            initializer=tf.random_normal_initializer(0, 0.3),
	                                            dtype=tf.float64)

			self.W = variable_scope.get_variable("context_W",
	                                            [context_size, encoder_size],
	                                            initializer=tf.random_normal_initializer(0, 0.3),
	                                            dtype=tf.float64)

			self.V1 = variable_scope.get_variable("context_v1",
	                                            [1, context_size],
	                                            initializer=tf.random_normal_initializer(0, 0.3),
	                                            dtype=tf.float64)

			self.last_context = hstates[:, -1, :]

			# (N x T x enc_size) x (enc_size x context_size) = (N x T x context_size)
			# To create such a product we resize hstates first do matmul and resize again
			F = tf.reshape(hstates, [-1, encoder_size])
			self.WF = tf.matmul(F, tf.transpose(self.W, [1,0]))
			self.WF = tf.reshape(self.WF, [-1, tf.shape(hstates)[1], context_size])
			# WF is a (N x T x context_size) tensor that the attention model in the decoder will work on.


class Decoder(object):
	"""
		Function of the decoder is to just do decoding given the context vector right.
		Implement RNN that just performs this decoding, again you have to check for embeddings if necessary.
	"""
	def __init__(self, graph, context, target_vocab_size, encoder_size, state_size,
				embed_size, num_layers, max_length, embedding, cell_type='LSTM'):
		
		self.graph = graph
		self.embedding = embedding

		with self.graph.as_default():
			# Variable definitions
			self.cell = _create_cell(state_size, num_layers, cell_type)
			self.w_out = variable_scope.get_variable("w_out",
	                                            [state_size, target_vocab_size],
	                                            initializer=tf.random_normal_initializer(0, 0.3),
	                                            dtype=tf.float64)
			self.b_out = variable_scope.get_variable("b_out",
	                                            [target_vocab_size],
	                                            initializer=tf.random_normal_initializer(0, 0.3),
	                                            dtype=tf.float64)

			self.decoder_inputs = tf.placeholder(tf.int32, [None, max_length], name='dec_inputs')
			self.decoder_lengths = tf.placeholder(tf.int32, [None], name='dec_lengths')
			self.decoder_outputs = tf.placeholder(tf.float32, [None, max_length], name='dec_outputs')

			# Creating embeddings if needed
			if self.embedding:
				#Create embedding lookup function for the entire batch
				self.embed_inputs = []
				for t in xrange(max_length):
					decoder_inp = self.decoder_inputs[:, t]
					self.embed_inputs.append(tf.cast(tf.nn.embedding_lookup(params=self.embedding, ids=decoder_inp), tf.float64)) 
				# Transpose the time axis so we have shape NxTxD tensor
				self.embed_inputs = tf.transpose(tf.stack(self.embed_inputs), perm=[1,0,2])
			else:
				self.embed_inputs = tf.cast(self.decoder_inputs, tf.float64)
				# Need to reshape to work with dynamic_rnn input
				self.embed_inputs = tf.reshape(self.embed_inputs, [-1, max_length, 1])
				embed_size = 1

			batch_size = tf.shape(self.embed_inputs)[0]

			def attention(prev_state, inp):
				# Creates the summary from the context
				# Returns the input concatenated with the summary
				summary = context.last_context
				# TODO implement attention based summary
				return tf.concat([summary, inp], 1)

			def last_state(prev_state, inp):
				# Creates the summary from the context
				# Returns the input concatenated with the summary
				summary = context.last_context
				return tf.concat([summary, inp], 1)

			def train_fn(time, cell_output, cell_state, loop_state):
			    emit_output = cell_output  # == None for time == 0
			    if cell_output is None:  # time == 0
					next_cell_state = self.cell.zero_state(batch_size, tf.float64)
			    else:
					next_cell_state = cell_state
			    elements_finished = (time >= self.decoder_lengths) # check which all batches finished processing input
			    finished = tf.reduce_all(elements_finished)
			    # This condition ensures that based on the input_length decoding is done.
			    next_input = tf.cond(
								finished, # if all the inputs in the batch are over
								lambda: tf.zeros([batch_size, embed_size + encoder_size], dtype=tf.float64), # when we are all done return 0 state
								lambda: last_state(next_cell_state, self.embed_inputs[:, time, :])) #  concatenate input vector to the input
			    next_loop_state = None
			    return (elements_finished, next_input, next_cell_state,
						emit_output, next_loop_state)

			outputs_ta, final_state, _ = tf.nn.raw_rnn(self.cell, train_fn)
			self.dec_states = outputs_ta.stack()

			# Training the network based on the output of decoder
			self.outputs = []
			losses = []
			
			for t in xrange(max_length):
				output = tf.matmul(self.dec_states[:, t, :], self.w_out) + self.b_out
				# Need to convert this to probabilities

				loss = tf.nn.sampled_softmax_loss(tf.cast(tf.transpose(self.w_out), tf.float32),
												tf.cast(self.b_out, tf.float32),
												tf.reshape(self.decoder_outputs[:, t], [-1, 1]),
												tf.cast(self.dec_states[:, t, :], tf.float32),
												num_sampled=1000,
												num_classes=target_vocab_size,
												num_true=1)

				losses.append(loss)
				self.outputs.append(output)

			self.outputs = tf.stack(self.outputs) # N x T x source_vocab
			losses = tf.stack(losses) # N x T

			# Mask the losses that don't carry meaning and average over those losses that carry meaning
			mask = tf.sequence_mask(self.decoder_lengths, max_length) # N x T
			losses = losses * tf.cast(mask, tf.float32)
			self.total_avg_loss = tf.cast(tf.reduce_sum(losses), tf.float64) / tf.reduce_sum(tf.cast(self.decoder_lengths, tf.float64))

			# TODO: Functions that perform beam_decode and greed_decode have to be defined here


	def train_step(self, session, dec_inputs, dec_lengths, dec_outputs):

		feed_dict = {self.decoder_inputs: dec_inputs,
					self.dec_lengths: dec_lengths,
					self.dec_outputs: dec_outputs}

		loss, _ = session.run([self.total_avg_loss, train_step], feed_dict)
		return loss
	

# Function returns the correct cell based on cell type
def _create_cell(state_size, num_layers, cell_type):
	def single_cell(state_size, cell_type):
		if cell_type == 'LSTM':
			cell = tf.contrib.rnn.BasicLSTMCell(state_size, state_is_tuple=True)
		elif cell_type == 'GRU':
			cell = tf.contrib.rnn.GRUCell(state_size, state_is_tuple=True)
		return cell
	# Increases the number of LSTMs accordingly
	if num_layers > 1:
		cell = tf.contrib.rnn.MultiRNNCell([_single_cell(state_size, cell_type) for _ in xrange(num_layers)])
	else:
		cell = single_cell(state_size, cell_type)
	return cell


if __name__ == '__main__':
	state_size = 100
	num_layers = 1
	cell_type = 'LSTM'
	embed_size = 200
	max_length = 40
	initial_state = np.zeros((2, state_size))
	source_vocab_size = 20
	X = np.random.randn(2, max_length)
	X = X.astype(np.int32) + 5
	decoder_size = 200

	# The second example is of length 6 
	X[1,6:] = 0
	X_lengths = [10, 6]
	graph = tf.Graph()
	with tf.Session(graph=graph) as sess:
		with tf.name_scope('Encoder') as main_scope:
			enc = Encoder(graph, source_vocab_size, state_size, num_layers, max_length, cell_type, embed_size)
			context = Context(graph, enc.enc_states, state_size, state_size, state_size)
			init_op = tf.global_variables_initializer()
			sess.run(init_op)
			print enc