import numpy as np
import collections
import os
import re
import csv
import sys
import _pickle as cPickle
from tensorflow.python.platform import gfile
import math
import json
try:
	import cPickle as pickle
except ImportError:
	import pickle

min_count = 3
add_count = 5

class BeamLink(object):
	def __init__(self, value, h_value):
		self.prev = None
		self.hash_value = h_value
		self.value = value
	def set_prev(self, prev):
		self.prev = prev

	def get_path(self):
		path = []
		c_prev = self
		while c_prev is not None:
			path.append(c_prev.value)
			c_prev = c_prev.prev
		path.reverse()
		return path

class Data(object):
	def __init__(self, feats, encoder_in_idx, decoder_in, v_encoder_in, truth_captions, t_encoder_in, files):
		self.length = encoder_in_idx.shape[0]
		self.current = 0
		self.feats = feats
		self.encoder_in_idx = encoder_in_idx
		self.decoder_in = decoder_in
		# validation data
		self.v_encoder_in = v_encoder_in
		self.truth_captions = truth_captions
		self.t_encoder_in = t_encoder_in
		self.files = files

	def next_batch(self, size):
		if self.current == 0:
			index = np.random.permutation(np.arange(self.length))
			self.encoder_in_idx = self.encoder_in_idx[index]
			self.decoder_in = self.decoder_in[index]

		if self.current + size < self.length:
			e_in_idx, d_in = self.encoder_in_idx[self.current:self.current+size], self.decoder_in[self.current:self.current+size,:-1]
			d_out = self.decoder_in[self.current:self.current+size,1:]
			e_in = self.feats[e_in_idx]
			self.current += size
		else:
			e_in_idx, d_in = self.encoder_in_idx[self.current:], self.decoder_in[self.current:,:-1]
			d_out = self.decoder_in[self.current:,1:]
			e_in = self.feats[e_in_idx]
			self.current = 0

		return e_in, d_in, d_out


class VocabularyProcessor(object):
	def __init__(self, max_document_length, vocabulary, unknown_limit=float('Inf'), drop=False):
		self.max_document_length = max_document_length
		self._reverse_mapping = ['<UNK>'] + vocabulary
		self.make_mapping()
		self.unknown_limit = unknown_limit
		self.drop = drop

	def make_mapping(self):
		self._mapping = {}
		for i, vocab in enumerate(self._reverse_mapping):
			self._mapping[vocab] = i

	def transform(self, raw_documents):
		data = []
		lengths = []
		for tokens in raw_documents:
			word_ids = np.ones(self.max_document_length, np.int32) * self._mapping['<EOS>']
			length = 0
			unknown = 0
			if self.drop and len(tokens.split()) > self.max_document_length:
				continue
			for idx, token in enumerate(tokens.split()):
				if idx >= self.max_document_length:
					break
				word_ids[idx] = self._mapping.get(token, 0)
				length = idx
				if word_ids[idx] == 0:
					unknown += 1
			length = length+1
			if unknown <= self.unknown_limit:
				data.append(word_ids)
				lengths.append(length)

		data = np.array(data)
		lengths = np.array(lengths)

		return data, lengths
			# yield word_ids
	def save(self, filename):
		with gfile.Open(filename, 'wb') as f:
			f.write(pickle.dumps(self))
	@classmethod
	def restore(cls, filename):
		with gfile.Open(filename, 'rb') as f:
			return pickle.loads(f.read())

def clean_str(string):
	string = re.sub(r"\.", r"", string)
	return string.strip().lower()

# load and dump the mapped train and valid's captions
def load_text_data(train_lab, prepro_train_p, vocab_path):
	tlab = json.load(open(train_lab, 'r'))
	vocab_dict = collections.defaultdict(int)
	train_dict = {}

	for caps in tlab:
		train_dict[caps['id']] = ['<BOS> '+clean_str(cap)+' <EOS>' for cap in caps['caption']]

	# build vocabulary
	maxlen = 0
	avglen = 0
	total_seq = 0
	for cid, captions in train_dict.items():
		for caption in captions:
			s_caption = caption.split()
			avglen += len(s_caption)
			total_seq += 1
			if len(s_caption) >= maxlen:
				maxlen = len(s_caption)

			for word in s_caption:
				vocab_dict[word] += 1
	vocabulary = []
	for k, v in sorted(vocab_dict.items(), key=lambda x:x[1], reverse=True):
		if v >= min_count:
			vocabulary.append(k)
	
	# map sequence to its id
	vocab_processor = VocabularyProcessor(max_document_length=math.ceil(avglen/total_seq)+add_count, vocabulary=vocabulary, drop=True)

	t_number = 0
	min_t = float('Inf')
	avg_t = 0
	for cid, _ in train_dict.items():
		train_c_dat, lengths = vocab_processor.transform(train_dict[cid])
		train_dict[cid] = {'captions':train_c_dat, 'lengths':lengths}
		t_number += len(train_c_dat)
		if len(train_c_dat) < min_t:
			min_t = len(train_c_dat)

	cPickle.dump(train_dict, open(prepro_train_p, 'wb'))
	vocab_processor.save(vocab_path)

	print('init sequence number: {}'.format(total_seq))
	print('maximum sequence length: {}'.format(maxlen))
	print('average sequence length: {}'.format(avglen/total_seq))
	print('drop length: > {}'.format(math.ceil(avglen/total_seq)+add_count))
	print('remaining total train number: {}'.format(t_number))
	print('total video number: {}'.format(len(train_dict)))
	print('minimum train number: {} per video'.format(min_t))
	print('average train number: {} per video'.format(t_number//len(train_dict)))

	return vocab_processor, train_dict

def load_valid(valid_dir, valid_lab):
	vlab = json.load(open(valid_lab, 'r'))
	paths = []
	feats = []
	truth_captions = []
	for caps in vlab:
		feat_path = os.path.join(valid_dir, caps['id']+'.npy')
		paths.append(feat_path)
		truth_captions.append([clean_str(cap) for cap in caps['caption']])
	for path in paths:
		feat = np.load(path)
		feats.append(feat)

	return np.array(feats, dtype='float32'), truth_captions

def load_task(task_dir):

	feats = []
	paths = []
	files = []
	for dirPath, dirNames, fileNames in os.walk(task_dir):
		for f in fileNames:
			paths.append(os.path.join(task_dir, f))
			files.append(f)
	for path in paths:
		feat = np.load(path)
		feats.append(feat)

	return np.array(feats, dtype='float32'), files

def gen_train_data(train_dir, train_lab, train_dict):
	tlab = json.load(open(train_lab, 'r'))
	paths = []
	feats = []
	for caps in tlab:
		feat_path = os.path.join(train_dir, caps['id']+'.npy')
		paths.append(feat_path)
	for path in paths:
		feat = np.load(path)
		feats.append(feat)

	# here instead of using feat value, we use idx of feat to indicate the feat (faster)
	encoder_in_idx = []
	decoder_in = []
	for idx, caps in enumerate(tlab):
		for d_f in train_dict[caps['id']]['captions']:
			encoder_in_idx.append(idx)
			decoder_in.append(d_f)
	encoder_in_idx = np.array(encoder_in_idx, dtype='int32')
	decoder_in = np.array(decoder_in, dtype='float32')
		
	return np.array(feats) ,encoder_in_idx, decoder_in

def get_unknown_word_vec(dim_size):
	return np.random.uniform(-0.25, 0.25, dim_size) 

def build_w2v_matrix(vocab_processor, w2v_path, vector_path, dim_size):
	w2v_dict = {}
	f = open(vector_path, 'r')
	for line in f.readlines():
		word, vec = line.strip().split(' ', 1)
		w2v_dict[word] = np.loadtxt([vec], dtype='float32')

	vocab_list = vocab_processor._reverse_mapping
	w2v_W = np.zeros(shape=(len(vocab_list), dim_size), dtype='float32')

	for i, vocab in enumerate(vocab_list):
		# unknown vocab
		if i == 0:
			continue
		else:
			if vocab in w2v_dict:
				w2v_W[i] = w2v_dict[vocab]
			else:
				w2v_W[i] = get_unknown_word_vec(dim_size)

	cPickle.dump(w2v_W, open(w2v_path, 'wb'))

	return w2v_W










