import numpy as np
import argparse, math, time, os, librosa
import chainer, codecs
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList, link
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from loader import SpeechLoader
from model import SpeechModel
from utils import printc, printb, printr

# test_wav_files = ['A4_0', 'A4_1', 'A4_3', 'A4_4']

def _get_sentence(y_batch, BLANK, vocab_token_to_id, vocab_id_to_token):
	sum_error = 0

	for batch_idx, argmax_sequence in enumerate(y_batch):
		pred_id_seqence = []
		prev_token = BLANK
		for token_id in argmax_sequence:
			if token_id == BLANK:
				prev_token = BLANK
				continue
			if token_id == prev_token:
				continue
			pred_id_seqence.append(int(token_id))
			prev_token = token_id

		pred_sentence = ""
		for token_id in pred_id_seqence:
			pred_sentence += vocab_id_to_token[token_id]

	return pred_sentence

def handle_file(dirpath, filename):
	if filename.endswith('.wav') or filename.endswith('.WAV'):
		filename_path = os.path.join(dirpath, filename)
		if os.stat(filename_path).st_size < 24000:
			return
		return filename_path

def get_test_wav_files(test_path):
	wav_files = []
	if test_path:
		for (dirpath, dirnames, filenames) in os.walk(test_path):
			for filename in filenames:
				if handle_file(dirpath,filename):
					wav_files.append(handle_file(dirpath,filename))
	return wav_files

# 语音识别
# 把batch_size改为1
def speech_to_text():
	n_mfcc = 60

	# 加载数据
	wav_path = '/home/czifan/Desktop/speech/data/czf_test'
	label_file = '/home/czifan/Desktop/speech/data/czf_test/test.word.txt'
	wav_path = '/home/czifan/Desktop/speech/data/czf_train'
	label_file = '/home/czifan/Desktop/speech/data/czf_train/train.word.txt'
	speech_loader = SpeechLoader(wav_path, label_file, batch_size=1, n_mfcc=60)


	# 加载模型
	model = SpeechModel(60, speech_loader.vocab_size)
	serializers.load_npz(os.path.join(os.getcwd(), 'my_model.model'), model)

	# 标签文件
	labels_dict = {}
	with codecs.open(label_file, "r", encoding="utf-8") as f:
		for label in f:
			label = label.strip('\n')
			labels_id = label.split(' ',1)[0]
			labels_text = label.split(' ',1)[1]
			labels_dict[labels_id] = labels_text
	test_wav_files = get_test_wav_files(wav_path)[:20]
	for wav_file in test_wav_files:
		wav_id = os.path.basename(wav_file).split('.')[0]
		if wav_id not in labels_dict:
			continue;
		label_text = labels_dict[wav_id].replace(' ', '')
		wav_file = os.path.join(wav_path, wav_file)
		wav, sr = librosa.load(wav_file, mono=True)
		mfcc = np.transpose(np.expand_dims(librosa.feature.mfcc(wav, sr, n_mfcc=n_mfcc), axis=0), [0,2,1])
		mfcc = mfcc.tolist()

		# fill 0
		while len(mfcc[0]) < speech_loader.wav_max_len:
			mfcc[0].append([0] * n_mfcc)

		# 字典
		wmap = {value:key for key, value in speech_loader.wordmap.items()}

		y_batch = model(np.array(mfcc, dtype=np.float32))
		y_batch = np.argmax(y_batch.data, axis=1)

		print("---------------------------")
		print("Input: " + wav_file)
		print("Output: " + _get_sentence(y_batch, speech_loader.get_blank_symbol(), speech_loader.wordmap.items(), wmap))
		print("Except: " + label_text)

if __name__ == '__main__':
	speech_to_text()