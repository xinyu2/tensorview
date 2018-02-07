###==========================================================
###==========================================================
### visualize the cnv-activations of CNN in training 
### reNN: load and restore CNN structure
###==========================================================
###==========================================================
import numpy as np
import tensorflow as tf

def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  w = tf.Variable(initial)
  return w

def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  b=tf.Variable(initial)
  return b

def load_variable(initial):
  """weight_variable generates a weight variable of a given shape."""
  x = tf.Variable(initial)
  return x

# Store layers weight & bias
def initNN(cnv1size,cnv2size):
    weights = {
        'k1':  weight_variable([5, 5, 1, cnv1size]),
        'k2':  weight_variable([5, 5, cnv1size, cnv2size]),
        'f1w': weight_variable([7 * 7 * cnv2size, 256]),
        'f2w': weight_variable([256, 10])
    }
    biases = {
        'b1': bias_variable([cnv1size]),
        'b2': bias_variable([cnv2size]),
        'f1b': bias_variable([256]),
        'f2b': bias_variable([10])
    }
    return weights,biases

def restoreNN(sw,sb):
    weights = {
        'k1':  load_variable(sw['k1']),
        'k2':  load_variable(sw['k2']),
        'f1w': load_variable(sw['f1w']),
        'f2w': load_variable(sw['f2w'])
    }
    biases = {
        'b1': load_variable(sb['b1']),
        'b2': load_variable(sb['b2']),
        'f1b': load_variable(sb['f1b']),
        'f2b': load_variable(sb['f2b'])
    }
    return weights,biases
