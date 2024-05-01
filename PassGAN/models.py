
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv1d

def ResBlock(name, inputs, dim):                                                            # INPUT ---------- |  INPUT Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = inputs                                                                         # Output           | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = tf.nn.relu(output)                                                             # ReLU             | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)                          # 1D Convolution   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = tf.nn.relu(output)                                                             # ReLU             | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)                          # 1D Convolution   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    return inputs + (0.3*output)                                                            # OUTPUT --------- | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)


def ResBlock(name, inputs, dim):                                                            # INPUT ---------- |  INPUT SHAPE = (BATCH_SIZE, 128, PW_LENGTH)
    output = inputs                                                                         # Output           | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = tf.nn.relu(output)                                                             # ReLU             | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = lib.ops.conv1d.Conv1D(name+'.1', dim, dim, 5, output)                          # 1D Convolution   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = tf.nn.relu(output)                                                             # ReLU             | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = lib.ops.conv1d.Conv1D(name+'.2', dim, dim, 5, output)                          # 1D Convolution   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    return inputs + (0.3*output)                                                            # OUTPUT --------- | OUTPUT SHAPE = (BATCH_SIZE, 128, PW_LENGTH)
   
    
def Generator(n_samples, seq_len, layer_dim, output_dim, prev_outputs=None):                # INPUT ---------- |  INPUT SHAPE = (None)
    output = make_noise(shape=[n_samples, 128])                                             # Noise            | Output Shape = (BATCH_SIZE, 128)
    output = lib.ops.linear.Linear('Generator.Input', 128, seq_len * layer_dim, output)     # Linear           | Output Shape = (BATCH_SIZE, 1280)
    output = tf.reshape(output, [-1, layer_dim, seq_len])                                   # Reshape          | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Generator.1', output, layer_dim)                                     # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Generator.2', output, layer_dim)                                     # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Generator.3', output, layer_dim)                                     # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Generator.4', output, layer_dim)                                     # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Generator.5', output, layer_dim)                                     # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = lib.ops.conv1d.Conv1D('Generator.Output', layer_dim, output_dim, 1, output)    # 1D Convolution   | Output Shape = (BATCH_SIZE, CHARMAP_SIZE, PW_LENGTH)
    output = tf.transpose(output, [0, 2, 1])                                                # Transpose        | Output Shape = (BATCH_SIZE, PW_LENGTH, CHARMAP_SIZE)
    output = softmax(output, output_dim)                                                    # Softmax          | Output Shape = (BATCH_SIZE, PW_LENGTH, CHARMAP_SIZE)
    return output                                                                           # OUTPUT --------- | OUTPUT SHAPE = (BATCH_SIZE, PW_LENGTH, CHARMAP_SIZE)


def Discriminator(inputs, seq_len, layer_dim, input_dim):                                   # INPUT ---------- |  INPUT SHAPE = (BATCH_SIZE, PW_LENGTH, CHARMAP_SIZE)
    output = tf.transpose(inputs, [0,2,1])                                                  # Transpose        | Output Shape = (BATCH_SIZE, CHARMAP_SIZE, PW_LENGTH) 
    output = lib.ops.conv1d.Conv1D('Discriminator.Input', input_dim, layer_dim, 1, output)  # 1D Convolution   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Discriminator.1', output, layer_dim)                                 # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Discriminator.2', output, layer_dim)                                 # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Discriminator.3', output, layer_dim)                                 # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Discriminator.4', output, layer_dim)                                 # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = ResBlock('Discriminator.5', output, layer_dim)                                 # Residual Block   | Output Shape = (BATCH_SIZE, 128, PW_LENGTH)
    output = tf.reshape(output, [-1, seq_len * layer_dim])                                  # Reshape          | Output Shape = (BATCH_SIZE, 128 * PW_LENGTH)
    output = lib.ops.linear.Linear('Discriminator.Output', seq_len * layer_dim, 1, output)  # Linear           | Output Shape = (BATCH_SIZE, 1)
    return output                                                                           # OUTPUT --------- | OUTPUT SHAPE = (BATCH_SIZE, 1)                                                                          # OUTPUT --------- | Output Shape = (BATCH_SIZE, 1)

def softmax(logits, num_classes):
    return tf.reshape(
        tf.nn.softmax(
            tf.reshape(logits, [-1, num_classes])
        ),
        tf.shape(logits)
    )

def make_noise(shape):
    return tf.random_normal(shape)
