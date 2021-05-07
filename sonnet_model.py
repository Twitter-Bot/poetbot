'''
Translation of Deep-speare into modern Tensorflow/Keras

Import Notes:
- Spacy has tokenizer
- can pad sequences with Keras

Model Notes:
  Preprocessing
    Spacy nlp.Tokenizer() breaks up text into Tokens (in this case tokens are the smallest parts of the parent text, i.e. words or characters)
    The return of running nlp(text) is a Doc of Tokens
    Tok2Vec(Token) produces a vector for each Token in Doc based on the spacy Tok2Vec model

    Use spacy to tokenize (spacy - Tokenizer()) and vectorize word inputs(spacy - Tok2Vec())
    Use spacy to tokenize (spacy - Tokenizer()) and vectorize character inputs(spacy - Tok2Vec())
    Concat word and character embeddings (spacy - Doc.from_docs(doc1, doc2))
  Encoder
    Feed into Bidirectional LSTM (keras.layers.bidirectional(layer = keras.layers.lstm, *args))
    Feed into Selective Encoder (keras.layers.attention(*args))
  Decoder

  Output

'''
import tensorflow as tf
import numpy as np
import spacy as sp
import re

from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, Embedding,
# from tensorflow.keras.preprocessing.sequence import pad_sequences

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

wordPipeline = sp.load("en_core_web_sm", exclude=["tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"])
charPipeline = sp.load("en_core_web_sm", exclude=["tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"])

charPipeline.tokenizer = Tokenizer(charPipeline.vocab, infix_finditer=re.compile(r'''.''').finditer)

text = "A man tokenizes"

wordDoc = wordPipeline(text)
charDoc = charPipeline(text)

# inputTensor = tf.concat([wordDoc.tensor, charDoc.tensor], 2)

# print(inputTensor)

# for token in doc1:
#   print(token.text)

# print('******************')

# for token in doc2:
#   print(token.text)
