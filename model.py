import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import layers

"""
Basic attention-based caption generator using tensorflow API
"""
class CaptionGeneratorBasic(object):
	def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
				decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
				start_id, end_id):
		self.e_in = tf.placeholder(tf.float32, [None, encoder_in_length, encoder_in_size], name='encoder_in')
		self.d_in_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_in_idx')
		self.d_out_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_out_idx')
		self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

		with tf.device("/cpu:0"):
			if word2vec_weight != None:
				self.W = tf.Variable(word2vec_weight, name='W')
			else:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))

			self.d_in_em = tf.nn.embedding_lookup(self.W, self.d_in_idx)

		with tf.name_scope("encoder"):

			self.en_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

			init_state = self.en_cell.zero_state(tf.shape(self.e_in)[0], dtype=tf.float32) #batch_size

			self.en_outputs, self.en_states = tf.nn.dynamic_rnn(self.en_cell, 
														self.e_in,  
														sequence_length=tf.fill([tf.shape(self.e_in)[0]], encoder_in_length), 
														dtype=tf.float32, 
														initial_state=init_state, 
														scope='rnn_encoder')

		with tf.variable_scope("decoder") as scope:
			output_fn = lambda x:layers.linear(x, vocab_size, biases_initializer=tf.constant_initializer(0), scope=scope)

			self.de_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

			# attention
			attention_keys, attention_values, attention_score_fn, attention_construct_fn = tf.contrib.seq2seq.prepare_attention(
													attention_states=self.en_outputs,
													attention_option='bahdanau',
													num_units=hidden_size)

			dynamic_fn_train = tf.contrib.seq2seq.attention_decoder_fn_train(
					                                self.en_states,
					                                attention_keys=attention_keys,
					                                attention_values=attention_values,
					                                attention_score_fn=attention_score_fn,
					                                attention_construct_fn=attention_construct_fn)

			self.de_outputs, self.de_states, _= tf.contrib.seq2seq.dynamic_rnn_decoder(
            										cell=self.de_cell, 
                                                    decoder_fn=dynamic_fn_train, 
                                                    inputs=self.d_in_em,
                                                    sequence_length=tf.fill([tf.shape(self.e_in)[0]], decoder_in_length),
                                                    name='rnn_decoder')

			self.train_logit = output_fn(self.de_outputs)

			self.flatten_logit = tf.reshape(self.train_logit, [-1, vocab_size])
			self.flatten_y = tf.reshape(self.d_out_idx, [-1]) 

			dynamic_fn_inference = tf.contrib.seq2seq.attention_decoder_fn_inference(
													output_fn=output_fn,
					                                encoder_state=self.en_states,
					                                attention_keys=attention_keys,
					                                attention_values=attention_values,
					                                attention_score_fn=attention_score_fn,
					                                attention_construct_fn=attention_construct_fn,
					                                embeddings=self.W,
					                                start_of_sequence_id=start_id,
					                                end_of_sequence_id=end_id,
					                                maximum_length=decoder_in_length,
					                                num_decoder_symbols=vocab_size
					                                )
			scope.reuse_variables()

			self.de_outputs_infer, self.de_states_infer, _ = tf.contrib.seq2seq.dynamic_rnn_decoder(
            										cell=self.de_cell, 
                                                    decoder_fn=dynamic_fn_inference, 
                                                    name='decoder_inference')

		with tf.name_scope("Loss"):
			loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.flatten_logit, labels=self.flatten_y)
			
			self.cost = tf.identity(loss, name='cost') 

"""
Basic attention-based caption generator without using tensorflow API
"""
class CaptionGeneratorMyBasic(object):
	def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
				decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
				start_id, end_id):
		self.e_in = tf.placeholder(tf.float32, [None, encoder_in_length, encoder_in_size], name='encoder_in')
		self.d_in_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_in_idx')
		self.d_out_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_out_idx')
		self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

		with tf.device("/cpu:0"):
			if word2vec_weight != None:
				self.W = tf.Variable(word2vec_weight, name='W')
			else:
				self.W = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=0.1))

			self.d_in_em = tf.nn.embedding_lookup(self.W, self.d_in_idx)

		with tf.name_scope("encoder"):

			self.en_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

			init_state = self.en_cell.zero_state(tf.shape(self.e_in)[0], dtype=tf.float32) #batch_size

			self.en_outputs, self.en_states = tf.nn.dynamic_rnn(self.en_cell, 
														self.e_in,  
														sequence_length=tf.fill([tf.shape(self.e_in)[0]], encoder_in_length), 
														dtype=tf.float32, 
														initial_state=init_state, 
														scope='rnn_encoder')

		with tf.variable_scope("decoder") as scope:
			output_fn = lambda x:layers.linear(x, vocab_size, biases_initializer=tf.constant_initializer(0), scope=scope)

			self.de_cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=True)

			# attention
			attention_keys, attention_values, attention_score_fn, attention_construct_fn = tf.contrib.seq2seq.prepare_attention(
													attention_states=self.en_outputs,
													attention_option='bahdanau',
													num_units=hidden_size)
			outputs = []
			state = self.en_states
			cell_output = self.en_outputs[:,-1]

			for step in range(decoder_in_length):
				if step > 0: scope.reuse_variables()
				context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
				input_vector = tf.concat([self.d_in_em[:, step, :], context_vector], axis=-1)
				(cell_output, state) = self.de_cell(input_vector, state)
				outputs.append(cell_output)

			outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

			self.train_logit = output_fn(outputs)

			self.flatten_logit = tf.reshape(self.train_logit, [-1, vocab_size])
			self.flatten_y = tf.reshape(self.d_out_idx, [-1])

			# inference
			preds = []
			state = self.en_states
			cell_output = self.en_outputs[:,-1]
			word_vector = tf.nn.embedding_lookup(self.W, tf.fill([tf.shape(self.e_in)[0]], start_id))

			for step in range(decoder_in_length):

				scope.reuse_variables()

				context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
				input_vector = tf.concat([word_vector, context_vector], axis=-1)
				(cell_output, state) = self.de_cell(input_vector, state)

				logit = output_fn(cell_output)
				max_idx = tf.argmax(logit, axis=1)
				word_vector = tf.nn.embedding_lookup(self.W, max_idx)

				preds.append(logit)
			
			self.de_outputs_infer = tf.transpose(tf.stack(preds), [1, 0, 2])

		with tf.name_scope("Loss"):
			loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.flatten_logit, labels=self.flatten_y)
			
			self.cost = tf.identity(loss, name='cost') 




