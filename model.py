import tensorflow as tf
import numpy as np
import math
from tensorflow.contrib.layers.python.layers import layers

"""
Basic attention-based caption generator using tensorflow dynamic_rnn_decoder API
"""
class CaptionGeneratorBasic(object):
	def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
				decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
				start_id, end_id, Bk=5):
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
Basic attention-based caption generator without using tensorflow dynamic_rnn_decoder API
Add Beam Search Inference
"""
class CaptionGeneratorMyBasic(object):
	def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
				decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
				start_id, end_id, Bk=5):
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

			# greedy inference
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

			""" 
				Beam Search START
			"""

			# first prediction

			state = self.en_states
			cell_output = self.en_outputs[:,-1]
			word_vector = tf.nn.embedding_lookup(self.W, tf.fill([tf.shape(self.e_in)[0]], start_id))
			scope.reuse_variables()
			context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
			input_vector = tf.concat([word_vector, context_vector], axis=-1)
			(cell_output, state) = self.de_cell(input_vector, state) # LSTMtuple((?, 256), (?, 256))
			logit = output_fn(cell_output)
			softmax = tf.nn.softmax(logit)

			f_value, f_idx = tf.nn.top_k(softmax, k=Bk) # (?, Bk)
			max_word = tf.nn.embedding_lookup(self.W, f_idx) # (?, Bk, 300)
			
			max_word_step = []
			c_index = []
			final_c = None

			expand_idx = tf.reshape(tf.concat([tf.expand_dims(tf.reshape(f_idx, [-1, 1]), axis=1)]*Bk, axis=1), [-1, Bk])
			max_word_step.append(tf.reshape(expand_idx, [-1, Bk*Bk]))
			max_prob = tf.log(f_value) # (?, Bk)

			word_vector = tf.reshape(max_word, [-1, embedding_size]) # (?*Bk, 300)
			c_s = tf.reshape(tf.concat([tf.expand_dims(state[0], axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)
			h_s = tf.reshape(tf.concat([tf.expand_dims(state[1], axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)
			state = tf.contrib.rnn.LSTMStateTuple(c_s, h_s) # ((?*Bk, 256) (?*Bk, 256))
			cell_out = tf.reshape(tf.concat([tf.expand_dims(cell_output, axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)	

			def prob_fn(prob_his, new_prb, w_idx, k): # (?, Bk) (?*Bk, Bk) (?*Bk, Bk)
				prob_his_x_1 = tf.concat([tf.expand_dims(tf.reshape(prob_his, [-1, 1]), axis=1)]*Bk, axis=1) #(?*Bk, Bk, 1)
				prob_his = tf.reshape(prob_his_x_1, [-1, Bk*Bk]) # (?, 16)
				new_prb = tf.reshape(new_prb, [-1, Bk*Bk])
				w_idx = tf.reshape(w_idx, [-1, 1])

				curr_prob = tf.add(prob_his, tf.log(new_prb)) # (?, 16)

				next_prob, idx = tf.nn.top_k(curr_prob, k=k, sorted=False) # (?, Bk)
				
				alignment = tf.reshape(tf.range(0, (tf.shape(idx)[0])*Bk*Bk, Bk*Bk), [-1, 1])
				a_idx = tf.add(idx, alignment)
				flatten_idx = tf.reshape(a_idx, [-1])

				next_w_id = tf.nn.embedding_lookup(w_idx, flatten_idx)
				next_w_id = tf.reshape(next_w_id, [-1, k])

				return next_prob, idx, flatten_idx, next_w_id

			def gen_input_fn(next_w_id, cell_output, state, flat_idx): # (?, Bk) (?*Bk, 256) ((?*Bk, 256) (?*Bk, 256)) (?*16, 1)
				max_word = tf.nn.embedding_lookup(self.W, next_w_id)
				word_vector = tf.reshape(max_word, [-1, embedding_size])

				c_s = tf.reshape(tf.concat([tf.expand_dims(state[0], axis=1)]*Bk, axis=1), [-1, hidden_size])
				c_s = tf.nn.embedding_lookup(c_s, flat_idx)
				h_s = tf.reshape(tf.concat([tf.expand_dims(state[1], axis=1)]*Bk, axis=1), [-1, hidden_size])
				h_s = tf.nn.embedding_lookup(h_s, flat_idx)
				state = tf.contrib.rnn.LSTMStateTuple(c_s, h_s)
				cell_out = tf.reshape(tf.concat([tf.expand_dims(cell_output, axis=1)]*Bk, axis=1), [-1, hidden_size])
				cell_out = tf.nn.embedding_lookup(cell_out, flat_idx) # (?*Bk, 256)

				return word_vector, cell_out, state

			def gen_context_vector(cell_out):
				cell_out = tf.reshape(cell_out, [-1, Bk, hidden_size])
				cell_out_s = tf.split(cell_out, Bk, axis=1)
				context_vector = []
				for s in range(Bk):
					out = tf.reshape(cell_out_s[s], [-1, hidden_size])
					context_v_s = attention_construct_fn(out, attention_keys, attention_values)
					context_v_exp = tf.expand_dims(context_v_s, axis=1)
					context_vector.append(context_v_exp)

				context_v_concat = tf.concat(context_vector, axis=1)

				return tf.reshape(context_v_concat, [-1, hidden_size])

			for step in range(1, decoder_in_length):

				scope.reuse_variables()

				context_vector = gen_context_vector(cell_out)
				# context_vector = attention_construct_fn(cell_out, attention_keys, attention_values)
				input_vector = tf.concat([word_vector, context_vector], axis=-1)
				(cell_output, state) = self.de_cell(input_vector, state)

				logit = output_fn(cell_output)
				softmax = tf.nn.softmax(logit)

				f_value, f_idx = tf.nn.top_k(softmax, k=Bk, sorted=False)
				max_word_step.append(tf.reshape(f_idx, [-1, Bk*Bk])) # (?*Bk, Bk) -> (?, sa_data_utils*Bk)

				if step < decoder_in_length-1:

					(max_prob, c_idx, flat_idx, next_w_id) = prob_fn(max_prob, f_value, f_idx, Bk)

					c_index.append(c_idx)

					(word_vector, cell_out, state) = gen_input_fn(next_w_id, cell_output, state, flat_idx)

				else:

					(max_prob, c_idx, flat_idx, next_w_id) = prob_fn(max_prob, f_value, f_idx, 1)

					final_c = c_idx

			print(tf.stack(max_word_step))
			print(tf.stack(c_index))
			print(final_c)

			self.step_max_words = tf.transpose(tf.stack(max_word_step), [1, 0, 2], name="step_max_words") 
			self.chosen_idx = tf.transpose(tf.stack(c_index), [1, 0, 2], name="chosen_idx")
			self.chosen_idx_f = tf.identity(final_c, name="chosen_idx_f")

			""" 
				Beam Search END 
				max_word_step -> len
				c_index -> len - 2 

			"""

		with tf.name_scope("Loss"):
			loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.flatten_logit, labels=self.flatten_y)
			
			self.cost = tf.identity(loss, name='cost') 


"""
Schedule Sampling attention-based caption generator without using tensorflow dynamic_rnn_decoder API
Add Beam Search Inference
"""
class CaptionGeneratorSS(object):
	def __init__(self, hidden_size, vocab_size, encoder_in_size, encoder_in_length,
				decoder_in_length, word2vec_weight, embedding_size, neg_sample_num,
				start_id, end_id, Bk=5):
		self.e_in = tf.placeholder(tf.float32, [None, encoder_in_length, encoder_in_size], name='encoder_in')
		self.d_in_idx = tf.placeholder(tf.int64, [None, decoder_in_length], name='decoder_in_idx')
		self.d_out_idx = tf.placeholder(tf.int32, [None, decoder_in_length], name='decoder_out_idx')
		self.use_pred = tf.placeholder(tf.bool, [None, decoder_in_length], name='use_pred')
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
			logits = []
			state = self.en_states
			cell_output = self.en_outputs[:,-1]
			max_idx = self.d_in_idx[:, 0]

			# Schedule Sampling
			for step in range(decoder_in_length):
				if step == 0:
					word_vector = self.d_in_em[:, step, :]
				else:
					word_idx = tf.where(self.use_pred[:, step], max_idx, self.d_in_idx[:, step])
					word_vector = tf.nn.embedding_lookup(self.W, word_idx)

				if step > 0: scope.reuse_variables()
				context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
				input_vector = tf.concat([word_vector, context_vector], axis=-1)
				(cell_output, state) = self.de_cell(input_vector, state)

				logit = output_fn(cell_output)
				logits.append(logit)
				max_idx = tf.stop_gradient(tf.argmax(logit, axis=1))

				outputs.append(cell_output)

			outputs = tf.transpose(tf.stack(outputs), [1, 0, 2])

			self.train_logit = tf.transpose(tf.stack(logits), [1, 0, 2])

			self.flatten_logit = tf.reshape(self.train_logit, [-1, vocab_size])
			self.flatten_y = tf.reshape(self.d_out_idx, [-1])

			# greedy inference
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

			""" 
				Beam Search START
			"""

			# first prediction

			state = self.en_states
			cell_output = self.en_outputs[:,-1]
			word_vector = tf.nn.embedding_lookup(self.W, tf.fill([tf.shape(self.e_in)[0]], start_id))
			scope.reuse_variables()
			context_vector = attention_construct_fn(cell_output, attention_keys, attention_values)
			input_vector = tf.concat([word_vector, context_vector], axis=-1)
			(cell_output, state) = self.de_cell(input_vector, state) # LSTMtuple((?, 256), (?, 256))
			logit = output_fn(cell_output)
			softmax = tf.nn.softmax(logit)

			f_value, f_idx = tf.nn.top_k(softmax, k=Bk) # (?, Bk)
			max_word = tf.nn.embedding_lookup(self.W, f_idx) # (?, Bk, 300)
			
			max_word_step = []
			c_index = []
			final_c = None

			expand_idx = tf.reshape(tf.concat([tf.expand_dims(tf.reshape(f_idx, [-1, 1]), axis=1)]*Bk, axis=1), [-1, Bk])
			max_word_step.append(tf.reshape(expand_idx, [-1, Bk*Bk]))
			max_prob = tf.log(f_value) # (?, Bk)

			word_vector = tf.reshape(max_word, [-1, embedding_size]) # (?*Bk, 300)
			c_s = tf.reshape(tf.concat([tf.expand_dims(state[0], axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)
			h_s = tf.reshape(tf.concat([tf.expand_dims(state[1], axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)
			state = tf.contrib.rnn.LSTMStateTuple(c_s, h_s) # ((?*Bk, 256) (?*Bk, 256))
			cell_out = tf.reshape(tf.concat([tf.expand_dims(cell_output, axis=1)]*Bk, axis=1), [-1, hidden_size]) # (?*Bk, 256)	

			def prob_fn(prob_his, new_prb, w_idx, k): # (?, Bk) (?*Bk, Bk) (?*Bk, Bk)
				prob_his_x_1 = tf.concat([tf.expand_dims(tf.reshape(prob_his, [-1, 1]), axis=1)]*Bk, axis=1) #(?*Bk, Bk, 1)
				prob_his = tf.reshape(prob_his_x_1, [-1, Bk*Bk]) # (?, 16)
				new_prb = tf.reshape(new_prb, [-1, Bk*Bk])
				w_idx = tf.reshape(w_idx, [-1, 1])

				curr_prob = tf.add(prob_his, tf.log(new_prb)) # (?, 16)

				next_prob, idx = tf.nn.top_k(curr_prob, k=k, sorted=False) # (?, Bk)
				
				alignment = tf.reshape(tf.range(0, (tf.shape(idx)[0])*Bk*Bk, Bk*Bk), [-1, 1])
				a_idx = tf.add(idx, alignment)
				flatten_idx = tf.reshape(a_idx, [-1])

				next_w_id = tf.nn.embedding_lookup(w_idx, flatten_idx)
				next_w_id = tf.reshape(next_w_id, [-1, k])

				return next_prob, idx, flatten_idx, next_w_id

			def gen_input_fn(next_w_id, cell_output, state, flat_idx): # (?, Bk) (?*Bk, 256) ((?*Bk, 256) (?*Bk, 256)) (?*16, 1)
				max_word = tf.nn.embedding_lookup(self.W, next_w_id)
				word_vector = tf.reshape(max_word, [-1, embedding_size])

				c_s = tf.reshape(tf.concat([tf.expand_dims(state[0], axis=1)]*Bk, axis=1), [-1, hidden_size])
				c_s = tf.nn.embedding_lookup(c_s, flat_idx)
				h_s = tf.reshape(tf.concat([tf.expand_dims(state[1], axis=1)]*Bk, axis=1), [-1, hidden_size])
				h_s = tf.nn.embedding_lookup(h_s, flat_idx)
				state = tf.contrib.rnn.LSTMStateTuple(c_s, h_s)
				cell_out = tf.reshape(tf.concat([tf.expand_dims(cell_output, axis=1)]*Bk, axis=1), [-1, hidden_size])
				cell_out = tf.nn.embedding_lookup(cell_out, flat_idx) # (?*Bk, 256)

				return word_vector, cell_out, state

			def gen_context_vector(cell_out):
				cell_out = tf.reshape(cell_out, [-1, Bk, hidden_size])
				cell_out_s = tf.split(cell_out, Bk, axis=1)
				context_vector = []
				for s in range(Bk):
					out = tf.reshape(cell_out_s[s], [-1, hidden_size])
					context_v_s = attention_construct_fn(out, attention_keys, attention_values)
					context_v_exp = tf.expand_dims(context_v_s, axis=1)
					context_vector.append(context_v_exp)

				context_v_concat = tf.concat(context_vector, axis=1)

				return tf.reshape(context_v_concat, [-1, hidden_size])

			for step in range(1, decoder_in_length):

				scope.reuse_variables()

				context_vector = gen_context_vector(cell_out)
				# context_vector = attention_construct_fn(cell_out, attention_keys, attention_values)
				input_vector = tf.concat([word_vector, context_vector], axis=-1)
				(cell_output, state) = self.de_cell(input_vector, state)

				logit = output_fn(cell_output)
				softmax = tf.nn.softmax(logit)

				f_value, f_idx = tf.nn.top_k(softmax, k=Bk, sorted=False)
				max_word_step.append(tf.reshape(f_idx, [-1, Bk*Bk])) # (?*Bk, Bk) -> (?, sa_data_utils*Bk)

				if step < decoder_in_length-1:

					(max_prob, c_idx, flat_idx, next_w_id) = prob_fn(max_prob, f_value, f_idx, Bk)

					c_index.append(c_idx)

					(word_vector, cell_out, state) = gen_input_fn(next_w_id, cell_output, state, flat_idx)

				else:

					(max_prob, c_idx, flat_idx, next_w_id) = prob_fn(max_prob, f_value, f_idx, 1)

					final_c = c_idx

			print(tf.stack(max_word_step))
			print(tf.stack(c_index))
			print(final_c)

			self.step_max_words = tf.transpose(tf.stack(max_word_step), [1, 0, 2], name="step_max_words") 
			self.chosen_idx = tf.transpose(tf.stack(c_index), [1, 0, 2], name="chosen_idx")
			self.chosen_idx_f = tf.identity(final_c, name="chosen_idx_f")

			""" 
				Beam Search END 
				max_word_step -> len
				c_index -> len - 2 

			"""

		with tf.name_scope("Loss"):
			loss = tf.contrib.losses.sparse_softmax_cross_entropy(logits=self.flatten_logit, labels=self.flatten_y)
			
			self.cost = tf.identity(loss, name='cost') 






