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

from spacy.tokens import Doc

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dense, Dropout, Embedding, GRU, LSTM

## \/ \/ \/ NEEDED FOR cuDNN GPU ACCELERATION \/ \/ \/ ##
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# spaCY pre-processing pipeline
nlp = sp.load("en_core_web_sm", exclude=["tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"]) # Loading spaCY pipeline

text = "finna" # Test text
char = list(text) # Breakup text into individual characters

wordDoc = nlp(text) # Word Embeddings
charDoc = Doc(nlp.vocab, char) # Char Embeddings

# Model Parameters
rate = 0.3
batch_size = 32

print(wordDoc.tensor)
print()
print(nlp(text).tensor)

# Model Encoder
encoder = Sequential()
encoder.add(Bidirectional(LSTM(200, return_sequences=True), input_shape='some shit'))
encoder.add(Dropout(rate))
encoder.add(Attention(inputs=200, return_attention_scores=False))

# Encoder Ouput -> Math (Calculation for weighted hidden states) -> GRU -> Final Output

# Do some crazy math shit here

# Decoder Input is character embeddings and

# Model Decoder
# Character Encodings \/
decoder = Sequential()
decoder.add(Bidirectional(LSTM(inputs='', units=200, merge_mode='concat', return_sequences=True))) # Produces Character Encodings
decoder.add(Dropout(rate))
# Concate Character Encodings with Word Embeddings
decoder.add(Concatenate()) # Concat word embeddings with character encodings
# MASK HERE \/ \/ \/ \/
decoder.add(LSTM(inputs='', mask='', units=200, return_sequences=True))

# Model Output
output = Sequential()
output.add(GRU())
output.add(Dense())

