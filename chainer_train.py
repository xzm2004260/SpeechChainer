import numpy as np
import argparse, math, time, os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList, link
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions
from loader import SpeechLoader
from model import SpeechModel
from utils import printc, printb, printr

wav_path = '/home/czifan/Desktop/speech/data/czf_train'
label_file = '/home/czifan/Desktop/speech/data/czf_train/train.word.txt'

# 设置参数
def _set_Args():
	parser = argparse.ArgumentParser(description='SpeechChainer')
	parser.add_argument('--batchsize', '-b', type=int, default=4,
						help='Number of images in each mini-batch')
	parser.add_argument('--epoch', '-e', type=int, default=20,
						help='Number of sweeps over the dataset to train')
	parser.add_argument('--frequency', '-f', type=int, default=-1,
						help='Frequency of taking a snapshot')
	parser.add_argument('--gpu', '-g', type=int, default=-1,
						help='GPU ID (negative value indicates CPU)')
	parser.add_argument('--out', '-o', default='result',
						help='Directory to output the result')
	parser.add_argument('--resume', '-r', default='',
						help='Resume the training from snapshot')
	parser.add_argument('--noplot', dest='plot', action='store_false',
						help='Disable PlotReport extension')
	parser.add_argument('--nmfcc', '-n', type=int, default=60,
						help='Number of mfcc features')
	args = parser.parse_args()
	print('GPU: {}'.format(args.gpu))
	print('# Minibatch-size: {}'.format(args.batchsize))
	print('# epoch: {}'.format(args.epoch))
	print('# nmfcc: {}'.format(args.nmfcc))
	print('')

	return args

def train():
	args = _set_Args()

	speech_loader = SpeechLoader(wav_path, label_file, batch_size=args.batchsize, n_mfcc=args.nmfcc)
	n_out = speech_loader.vocab_size

	model = SpeechModel(args.nmfcc, n_out)

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	printb('[Training]')

	for epoch in range(args.epoch):
		speech_loader.create_batches()
		speech_loader.reset_batch_pointer()
		for batch in range(speech_loader.n_batches):
			start = time.time()
			# 获取数据
			batches_wav, batches_label = speech_loader.next_batch()
			print(np.shape(batches_wav))
			# ctc_loss
			x_length_batch = np.array([len(x_) for x_ in batches_wav], dtype=np.int32)
			t_length_batch = np.array([len(t_) for t_ in batches_label], dtype=np.int32)
			y_batch = model(batches_wav)
			y_batch = F.swapaxes(y_batch, 0, 1)
			y_batch = F.swapaxes(y_batch, 0, 2)
			loss = F.connectionist_temporal_classification(list(y_batch), batches_label, speech_loader.get_blank_symbol(), x_length_batch, t_length_batch)
			# 更新
			optimizer.update(lossfun=lambda: loss)
			end = time.time()
			print("epoch: %d/%d, batch: %d/%d, loss: %s, time: %.3f."%(epoch, args.epoch, batch, speech_loader.n_batches, loss.data, end-start))

			if batch%20 == 0:
				serializers.save_npz(os.path.join(os.getcwd(), 'model','speech'+str(epoch)+'.module'), model)

class CTCLoss(Chain):
	def __init__(self, predictor, id_blank):
		super(CTCLoss, self).__init__()
		self.id_blank = id_blank
		with self.init_scope():
			self.predictor = predictor

	def __call__(self, batches_wav, batches_label):
		# ctc_loss
		x_length_batch = np.array([len(x_) for x_ in batches_wav], dtype=np.int32)
		t_length_batch = np.array([len(t_) for t_ in batches_label], dtype=np.int32)
		y_batch = self.predictor(batches_wav)
		y_batch = F.swapaxes(y_batch, 0, 1)
		y_batch = F.swapaxes(y_batch, 0, 2)
		self.loss = F.connectionist_temporal_classification(list(y_batch), batches_label, self.id_blank, x_length_batch, t_length_batch)
		report({"loss": self.loss}, self)
		return self.loss

def chainer_train():
	args = _set_Args()

	speech_loader = SpeechLoader(wav_path, label_file, batch_size=args.batchsize, n_mfcc=args.nmfcc)
	n_out = speech_loader.vocab_size

	base_model = SpeechModel(args.nmfcc, n_out)
	serializers.load_npz(os.path.join(os.getcwd(), 'my_model.model'), base_model)
	model = CTCLoss(base_model, speech_loader.get_blank_symbol())

	optimizer = chainer.optimizers.Adam()
	optimizer.setup(model)

	batches = speech_loader.get_batches()

	train_iter = chainer.iterators.SerialIterator(batches, args.batchsize)
	# test_iter = chainer.iterators.SerialIterator(test_batches, args.batchsize, repeat=False, shuffle=False)
	'''构造trainer'''
	updater = training.StandardUpdater(train_iter, optimizer, device=args.gpu)
	trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
	# trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu))
	trainer.extend(extensions.dump_graph('main/loss'))
	frequency = args.epoch if args.frequency == -1 else max(1, args.frequency)
	trainer.extend(extensions.snapshot(), trigger=(frequency, 'epoch'))
	trainer.extend(extensions.LogReport())
	if args.plot and extensions.PlotReport.available():
		trainer.extend(extensions.PlotReport(['main/loss'], 'epoch', file_name='loss.png'))
	trainer.extend(extensions.PrintReport(
		['epoch', 'main/loss', 'elapsed_time']))

	if args.resume:
		chainer.serializers.load_npz(args.resume, trainer)

	'''开始训练'''
	trainer.run()

	'''保存训练好的模型'''
	serializers.save_npz('my_model.model', base_model)


if __name__ == '__main__':
	# train()
	chainer_train()

