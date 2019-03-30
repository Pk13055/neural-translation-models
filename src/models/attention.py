#!/usr/bin/env python3
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import LSTM, Dense, Embedding, RepeatVector, TimeDistributed
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.utils.vis_utils import plot_model
from numpy import array

from .utils import collect_args


# load a clean dataset
def load_clean_sentences(filename):
	return load(open(filename, 'rb'))

# fit a tokenizer
def create_tokenizer(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

# max sentence length
def max_length(lines):
	return max(len(line.split()) for line in lines)

# encode and pad sequences
def encode_sequences(tokenizer, length, lines):
	# integer encode sequences
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

# one hot encode target sequence
def encode_output(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = array(ylist)
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
	return y

# define NMT model
def define_model(src_vocab, tar_vocab, src_timesteps, tar_timesteps, n_units):
	model = Sequential()
	model.add(Embedding(src_vocab, n_units, input_length=src_timesteps, mask_zero=True))
	model.add(LSTM(n_units))
	model.add(RepeatVector(tar_timesteps))
	model.add(LSTM(n_units, return_sequences=True))
	model.add(TimeDistributed(Dense(tar_vocab, activation='softmax')))
	return model


def main():
	# load datasets
	args = collect_args()

	store = pd.HDFStore(args.data)
	train, test = store['train'], store['test']

	# prepare english tokenizer
	eng_tokenizer = create_tokenizer(train['en'])
	eng_vocab_size = len(eng_tokenizer.word_index) + 1
	eng_length = max_length(train['en'])
	print('English Vocabulary Size: %d' % eng_vocab_size)
	print('English Max Length: %d' % (eng_length))
	# prepare german tokenizer
	ger_tokenizer = create_tokenizer(dataset[:, 1])
	ger_vocab_size = len(ger_tokenizer.word_index) + 1
	ger_length = max_length(dataset[:, 1])
	print('German Vocabulary Size: %d' % ger_vocab_size)
	print('German Max Length: %d' % (ger_length))

	# prepare training data
	trainX = encode_sequences(ger_tokenizer, ger_length, train[:, 1])
	trainY = encode_sequences(eng_tokenizer, eng_length, train[:, 0])
	trainY = encode_output(trainY, eng_vocab_size)
	# prepare validation data
	testX = encode_sequences(ger_tokenizer, ger_length, test[:, 1])
	testY = encode_sequences(eng_tokenizer, eng_length, test[:, 0])
	testY = encode_output(testY, eng_vocab_size)

	# define model
	model = define_model(ger_vocab_size, eng_vocab_size, ger_length, eng_length, 256)
	model.compile(optimizer='adam', loss='categorical_crossentropy')
	# summarize defined model
	print(model.summary())
	plot_model(model, to_file='model.png', show_shapes=True)
	# fit model
	filename = 'model.h5'
	checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
	model.fit(trainX, trainY, epochs=30, batch_size=64, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)

if __name__ == "__main__":
	main()
