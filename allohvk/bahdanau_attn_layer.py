import tensorflow as tf

class Bahdanau(tf.keras.layers.Layer):

    def __init__(self, n):
        super(Bahdanau, self).__init__()
        self.w = tf.keras.layers.Dense(n)
        self.u = tf.keras.layers.Dense(n)
        self.v = tf.keras.layers.Dense(1)

    def call(self, stminus1, h):
        stminus1 = tf.expand_dims(stminus1, 1)
        e = self.v(tf.nn.tanh(self.w(stminus1) + self.u(h)))
        a = tf.nn.softmax(e, axis=1)
        c = a * h
        c = tf.reduce_sum(c, axis=1)
        return a,c
