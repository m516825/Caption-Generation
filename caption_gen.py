import tensorflow as tf
import numpy as np
import os
import progressbar
import sys
import time
import data_utils
import _pickle as cPickle
from data_utils import VocabularyProcessor
from data_utils import Data
from model import CaptionGeneratorBasic, CaptionGeneratorMyBasic
import progressbar as pb
# import bleu_eval as BLEU
import BLEU as my_BLEU

tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of word embedding (default: 300)")
tf.flags.DEFINE_integer("hidden", 256, "hidden dimension of RNN hidden size")
tf.flags.DEFINE_integer("epoch", 100, "number of training epoch")
tf.flags.DEFINE_integer("batch_size", 100, "batch size per iteration")
tf.flags.DEFINE_integer("predict_every", 200, "predict model on dev set after this many steps (default: 200)")
tf.flags.DEFINE_integer("checkpoint_every", 200, "Save model after this many steps (default: 200)")
tf.flags.DEFINE_integer("dev_size", 200, "dev size")
tf.flags.DEFINE_integer("num_sampled", 500, "number of negative sampling")

tf.flags.DEFINE_float("lr", 1e-2, "training learning rate")
tf.flags.DEFINE_float("dropout_keep_prob", 0.7, "drop out rate")

tf.flags.DEFINE_string("data_dir", "./MLDS_hw2_data/", "Data path")
tf.flags.DEFINE_string("train_dir", "./MLDS_hw2_data/training_data/feat/", "train data directory with feat feature")
tf.flags.DEFINE_string("train_lab", "./MLDS_hw2_data/training_label.json", "train data, id and caption")
tf.flags.DEFINE_string("test_id", "./MLDS_hw2_data/testing_id.txt", "testing data id")
tf.flags.DEFINE_string("test_dir", "./MLDS_hw2_data/testing_data/feat/", "test data directory with feat feature")
tf.flags.DEFINE_string("valid_lab", "./MLDS_hw2_data/testing_public_label.json", "validation data, id and caption")
tf.flags.DEFINE_string("valid_dir", "./MLDS_hw2_data/testing_data/feat/", "validation data directory with feat feature")
tf.flags.DEFINE_string("vector_file", "./MLDS_hw2_data/glove/glove.6B.300d.txt", "Word representation vectors' file")
tf.flags.DEFINE_string("checkpoint_file", "", "checkpoint_file to be load")
tf.flags.DEFINE_string("w2v_data", "./prepro/w2v_W.dat", "word to vector matrix for our vocabulary")
tf.flags.DEFINE_string("prepro_train", "./prepro/train.dat", "tokenized train data's path")
tf.flags.DEFINE_string("vocab", "./vocab", "vocab processor path")
tf.flags.DEFINE_string("output", "./pred.txt", "output file")

tf.flags.DEFINE_boolean("eval", False, "Evaluate testing data")
tf.flags.DEFINE_boolean("prepro", True, "To do the preprocessing")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

class CapGenModel(object):
	def __init__(self, data, w2v_W, vocab_processor, use_nce=True):
		config = tf.ConfigProto(allow_soft_placement = True)
		config.gpu_options.allow_growth = True
		self.sess = tf.Session(config = config)
		self.data = data
		self.w2v_W = w2v_W
		self.vocab_processor = vocab_processor
		self.vocab_size = len(vocab_processor._reverse_mapping)
		self.sample_num = FLAGS.num_sampled if use_nce else None
		self.gen_path()

	def build_model(self):
		self.model = CaptionGeneratorBasic(hidden_size=FLAGS.hidden, 
									vocab_size=self.vocab_size, 
									encoder_in_size=self.data.feats.shape[-1], 
									encoder_in_length=self.data.feats.shape[1],
									decoder_in_length=self.data.decoder_in.shape[-1] - 1, 
									word2vec_weight=self.w2v_W,
									embedding_size=FLAGS.embedding_dim,
									neg_sample_num=self.sample_num,
									start_id=self.vocab_processor._mapping['<BOS>'],
									end_id=self.vocab_processor._mapping['<EOS>'])
		self.global_step = tf.Variable(0, name='global_step', trainable=False)

		self.optimizer = tf.train.AdamOptimizer(FLAGS.lr)

		tvars = tf.trainable_variables()

		grads, _ = tf.clip_by_global_norm(tf.gradients(self.model.cost, tvars), 5)

		self.updates = self.optimizer.apply_gradients(
						zip(grads, tvars), global_step=self.global_step)
		self.saver = tf.train.Saver(tf.global_variables())

	def gen_path(self):
		# Output directory for models and summaries
		timestamp = str(time.strftime('%b-%d-%Y-%H-%M-%S'))
		self.out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
		print ("Writing to {}\n".format(self.out_dir))
	    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
		self.checkpoint_dir = os.path.abspath(os.path.join(self.out_dir, "checkpoints"))
		self.checkpoint_prefix = os.path.join(self.checkpoint_dir, "model")
		if not os.path.exists(self.checkpoint_dir):
			os.makedirs(self.checkpoint_dir)
	
	def train(self):
		batch_num = self.data.length//FLAGS.batch_size if self.data.length%FLAGS.batch_size==0 else self.data.length//FLAGS.batch_size + 1
		current_step = 0
		with self.sess.as_default():
			if FLAGS.checkpoint_file == "":
				self.sess.run(tf.global_variables_initializer())
			else:
				self.saver.restore(sess, FLAGS.checkpoint_file)

			for ep in range(FLAGS.epoch):
				cost = 0.
				pbar = pb.ProgressBar(widget=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=batch_num).start()
				print("Epoch {}".format(ep+1))

				for b in range(batch_num):

					e_in, d_in_idx, d_out_idx = self.data.next_batch(FLAGS.batch_size)

					feed_dict = {
						self.model.e_in:e_in,
						self.model.d_in_idx:d_in_idx,
						self.model.d_out_idx:d_out_idx	
					}

					loss, step, _ = self.sess.run([self.model.cost, self.global_step, self.updates], feed_dict=feed_dict)

					current_step = tf.train.global_step(self.sess, self.global_step)

					cost += loss/batch_num

					pbar.update(b+1)
				pbar.finish()

				print (">>cost: {}".format(cost/batch_num))

				path = self.saver.save(self.sess, self.checkpoint_prefix, global_step=current_step)
				print ("\nSaved model checkpoint to {}\n".format(path))

				self.inference()

	def inference(self):
		feed_dict = {
				self.model.e_in:self.data.v_encoder_in,
			}
		preds = self.sess.run(self.model.de_outputs_infer, feed_dict=feed_dict)
		sentences = []

		for ps in preds:
			s = []
			for w in ps:
				idx = np.argmax(w)
				vocab = self.vocab_processor._reverse_mapping[idx]
				if vocab == '<EOS>':
					break
				s.append(vocab)
			sentences.append(' '.join(s))

		print(sentences[0])
		print(sentences[1])
		print(sentences[2])
		print(sentences[3])

		score = 0.
		nltk_score = 0.
		for idx, s in enumerate(sentences):
			# bleu = BLEU.BLEU_score([s], self.data.truth_captions[idx])
			bleu = my_BLEU.BLEU_score(s, self.data.truth_captions[idx])
			
			score += bleu

		print("BLEU score {}".format(score/len(sentences)))

def main(_):
	print("\nParameters: ")
	for k, v in sorted(FLAGS.__flags.items()):
		print("{} = {}".format(k, v))

	if not os.path.exists("./prepro/"):
		os.makedirs("./prepro/")

	if FLAGS.eval:
		print("Evaluation...")
	else:
		if FLAGS.prepro:
			print ("Start preprocessing data...")
			vocab_processor, train_dict = data_utils.load_text_data(train_lab=FLAGS.train_lab, 
														 prepro_train_p=FLAGS.prepro_train, vocab_path=FLAGS.vocab)
			print ("Vocabulary size: {}".format(len(vocab_processor._reverse_mapping)))
			
			print ("Start dumping word2vec matrix...")
			w2v_W = data_utils.build_w2v_matrix(vocab_processor, FLAGS.w2v_data, FLAGS.vector_file, FLAGS.embedding_dim)

		else:
			train_dict = cPickle.load(open(FLAGS.prepro_train, 'rb'))
			vocab_processor = VocabularyProcessor.restore(FLAGS.vocab)
			w2v_W = cPickle.load(open(FLAGS.w2v_data, 'rb'))

		print("Start generating training data...")
		feats, encoder_in_idx, decoder_in = data_utils.gen_train_data(FLAGS.train_dir, FLAGS.train_lab, train_dict)
		print("Start generating validation data...")
		v_encoder_in, truth_captions = data_utils.load_valid(FLAGS.valid_dir, FLAGS.valid_lab)

		print('feats size: {}, training size: {}'.format(len(feats), len(encoder_in_idx)))
		print(encoder_in_idx.shape, decoder_in.shape)
		print(v_encoder_in.shape, len(truth_captions))

		data = Data(feats, encoder_in_idx, decoder_in, v_encoder_in, truth_captions)

		model = CapGenModel(data, w2v_W, vocab_processor)

		model.build_model()

		model.train()
	
if __name__ == "__main__":
	tf.app.run()


