import tensorflow as tf
from keras import backend as K
from keras.layers import Flatten, Activation, RepeatVector, Permute, Multiply, Lambda

class peel_the_layer(tf.keras.layers.Layer):
    def __init__(self):
        # Nothing special to be done here
        super(peel_the_layer, self).__init__()

    def build(self, input_shape):
        # Define the shape of the weights and bias in this layer
        # As we discussed the layer has just 1 lonely neuron
        # We discussed the shapes of the weights and bias earlier
        self.w=self.add_weight(shape=(256,1), initializer="normal")
        self.b=self.add_weight(shape=(19,1), initializer="zeros")
        super(peel_the_layer, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x,self.w)+self.b)
        e = Flatten()(e)
        a = Activation('softmax')(e)
        # Don't manipulate 'a'. It needs to be 'return'ed intact
        temp = RepeatVector(256)(a)
        temp = Permute([2,1])(temp)

        output = Multiply()([x,temp])
        output = Lambda(lambda values: K.sum(values, axis=1))(output)

        return a, output
