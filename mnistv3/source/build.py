import tensorflow as tf
import numpy as np
from PIL import Image


model_saver_file = './cnn_mnist/model'

sess = tf.Session()

x_ = tf.placeholder(dtype=tf.float32, shape=[None, 784])
x_train = tf.reshape(x_, shape=[-1,28,28,1])
y_train = tf.placeholder(dtype=tf.float32, shape=[None, 10])

#Input layer
input_layer = x_train

# Convolutional Layer #1 + Pooling
# Input Tensor Shape: [batch_size, 28, 28, 1] kernel_size 5x5
# Output Tensor Shape: [batch_size, 28, 28, 32] -> pooling [batch_size , 14,14,32]
conv1 = tf.layers.conv2d(
    inputs=input_layer,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

# Convolutional Layer #2 + pooling
# Input Tensor Shape: [batch_size, 14, 14, 32] kernel_size 3x3
# Output Tensor Shape: [batch_size, 14, 14, 64] -> pooling [batch_size, 7,7,64]
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=tf.nn.relu)
pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

# Flatten tensor into a batch of vectors
# Input Tensor Shape: [batch_size, 7, 7, 64]
# Output Tensor Shape: [batch_size, 7 * 7 * 64]
pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

# FC layer1 - 1024 neurons
# Input Tensor Shape: [batch_size, 7 * 7 * 64]
# Output Tensor Shape: [batch_size, 1024]
dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)

# FC layer2 - Logits
# Input Tensor Shape: [batch_size, 1024]
# Output Tensor Shape: [batch_size, 10]
logits = tf.layers.dense(inputs=dense, units=10)
smlogits = tf.nn.softmax(logits)

saver = tf.train.Saver()

try:
    saver.restore(sess, model_saver_file)
    prediction = tf.argmax(logits, 1)
except Exception:
    print('Saver not found')

def test():
    try:
        pic = Image.open("./input.jpg").convert('L')
        xx = np.array(pic.getdata())
        for i in range(len(xx)):
            if (1 - xx[i] / 255) > 0.9:
                xx[i] = 1
            else:
                xx[i] = 0
        x_input = np.array([xx])

        ans,acc = sess.run((prediction,smlogits) , {x_: x_input, y_train: np.array([np.zeros(10)])})
        print('ans : ', ans)
        print('acc: ' , acc)
        return ans[0] , acc[0]

    except Exception:
        print('Saver not found')
