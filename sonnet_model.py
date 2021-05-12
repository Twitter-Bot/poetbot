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
# from tensorflow.keras.initializers import Constant
# from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
from tensorflow.keras.layers import Attention, Bidirectional, Concatenate, Dropout, Embedding, GRU, LSTM

## \/ \/ \/ NEEDED FOR cuDNN GPU ACCELERATION \/ \/ \/ ##
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)

# spaCY pre-processing pipeline
nlp = sp.load("en_core_web_sm", exclude=["tagger", "parser", "senter", "ner", "attribute_ruler", "lemmatizer"]) # Loading spaCY pipeline

text = "A man tokenizes" # Test text
char = list(text) # Breakup text into individual characters

wordDoc = nlp(text) # Word Embeddings
charDoc = Doc(nlp.vocab, char) # Char Embeddings

'''
Below is code from embedding article
'''

# Vectorizer = TextVectorization()

# #fit the vectorizer on the text and extract the corpus vocabulary
# Vectorizer.adapt(text.Text.to_numpy())
# vocab = Vectorizer.get_vocabulary()

# #generate the embedding matrix
# num_tokens = len(vocab)
# embedding_dim = len(wordDoc.vector)
# embedding_matrix = np.zeros((num_tokens, embedding_dim))
# for i, word in enumerate(vocab):
#   embedding_matrix[i] = nlp(word).vector

# #Load the embedding matrix as the weights matrix for the embedding layer and set trainable to False
# Embedding_layer=Embedding(
#   num_tokens,
#   embedding_dim,
#   embeddings_initializer=Constant(embedding_matrix),
#   trainable=False)

# Model Parameters
rate = 0.3
batch_size = 32

# Model Encoder
encoder = Sequential()
encoder.add(Bidirectional(LSTM(200, return_sequences=True), input_shape='some shit'))
encoder.add(Dropout(rate))
encoder.add(Attention(inputs=200, return_attention_scores=False))

# Do some crazy math shit here

# Model Decoder
decoder = Sequential()
decoder.add(Bidirectional(LSTM(inputs='', units=200, return_sequences=True)))
decoder.add(Dropout(rate))
decoder.add(Concatenate()) # Concat word embeddings with character encodings
# MASK HERE \/ \/ \/ \/
decoder.add(LSTM(inputs='', mask='', units=200, return_sequences=True))

# Model Output

# inputTensor = tf.concat([wordDoc.tensor, charDoc.tensor], 2)

# print(inputTensor)

# for token in doc1:
#   print(token.text)

# print('******************')

print(charDoc)
for token in charDoc:
  print(token.text)
