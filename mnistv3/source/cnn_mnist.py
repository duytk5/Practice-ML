import tensorflow as tf
import datetime

from tensorflow.examples.tutorials.mnist import input_data

model_saver_file = './cnn_mnist/model'

def main(argv):
    mnist = input_data.read_data_sets('MNIST-data/', one_hot= True)
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
    #loss
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_train, logits=logits))
    optimizer = tf.train.GradientDescentOptimizer(0.0003).minimize(cross_entropy)


    init = tf.global_variables_initializer()

    nsteps = 2000
    saver = tf.train.Saver()

    #train
    sess.run(init)
    print("Train : ")
    try:
        saver.restore(sess, model_saver_file)
    except Exception:
        print("Saving file doesn't exist")
    time1 = datetime.datetime.now()
    for step in range(1,nsteps+1):
        batch_x, batch_y = mnist.train.next_batch(100)
        for ii in range(0, len(batch_x)):
            for jj in range(0, len(batch_x[ii])):
                if (batch_x[ii][jj] >= 0.8):
                    batch_x[ii][jj] = 1
                else:
                    batch_x[ii][jj] = 0
        loss, _ = sess.run([cross_entropy, optimizer], {x_ : batch_x, y_train : batch_y})

        if(step % 500 == 0):
            time2 = datetime.datetime.now() ;
            time = (time2 - time1).total_seconds();
            time1 = datetime.datetime.now()
            print(time,'s')
            print('local step : ', step, ' ; loss = ', loss)
        if(step == nsteps):
            save_path = saver.save(sess, model_saver_file)
            print('Training Completed')
            print('Saving successfully, save path : ', save_path)

    #test

    try:
        saver.restore(sess, model_saver_file )
        prediction = tf.argmax(logits,1)
        true_label = tf.argmax(y_train,1)

        pre = tf.equal(prediction, true_label)
        accuracy = tf.reduce_mean(tf.cast(pre, tf.float32))

        x_input = mnist.test.images
        for ii in range(0, len(x_input)):
            for jj in range(0, len(x_input[ii])):
                if (x_input[ii][jj] >= 0.8):
                    x_input[ii][jj] = 1
                else:
                    x_input[ii][jj] = 0
        accuracy,ans = sess.run((accuracy,smlogits), {x_: x_input, y_train: mnist.test.labels})

        print('accuracy : ',accuracy)
    except Exception:
        print('Saver not found')

if __name__ == '__main__':
    tf.app.run()