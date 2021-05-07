import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding,
#can pad sequences with keras
#spacy has tokenizer
from keras.preprocessing.sequence import pad_sequences
import spacy
#decoder

#encoder
#bidrectional LSTM
#attention tensors
'''
we embed context words using embedding matrices to produce a sequence of
hidden states

'''
#output later
