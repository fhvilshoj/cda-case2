import sys
sys.path.append('.')

import numpy as np
import tensorflow as tf

from data_feed import Reader


class Data:
    def __init__(self, test_set=False):
        "docstring"
        np.random.seed(42)
        feed = Reader()
        images, labels = feed.data()
        selection = np.random.permutation(images.shape[0])
        train_size = int(images.shape[0] * 0.8)

        self.train_images = images[selection[:train_size]]
        self.train_labels = labels[selection[:train_size]]
        self.val_images = images[selection[train_size:]]
        self.val_labels = labels[selection[train_size:]]

        self.selection = np.random.permutation(self.train_images.shape[0])
                
        self.count = 0

    def next(self, batch_size=75):
        i = self.train_images[self.selection[self.count:self.count + batch_size]]
        l = self.train_labels[self.selection[self.count:self.count + batch_size]]

        self.count = self.count + batch_size
        if self.count > len(self.train_images) - batch_size - 1:
            self.selection = np.random.permutation(self.train_images.shape[0])
            self.count = 0
        return i, l
                
    def validation(self):
        return self.val_images, self.val_labels

def model(is_training):
    data = Data(not is_training)
    
    with tf.Graph().as_default() as graph:

        x = tf.placeholder(dtype=tf.float32, shape=(None, 170, 384))
        y_ = tf.placeholder(dtype=tf.float32, shape=(None,))
            
        training = tf.placeholder(dtype=tf.bool)
        print(x)
        x_reshaped = tf.reshape(x, shape=[-1, 170, 384, 1])

        conv1 = tf.layers.conv2d(inputs = x_reshaped, filters=8, kernel_size=5, padding='same')
        batch1 = tf.layers.batch_normalization(inputs=conv1, training=training)
        relu1 = tf.nn.relu(batch1)
        max_pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=3, strides=3, padding='same')
        
        conv2 = tf.layers.conv2d(inputs = max_pool1, filters=16, kernel_size=5, padding='same')
        batch2 = tf.layers.batch_normalization(inputs=conv2, training=training)
        relu2 = tf.nn.relu(batch2)
        max_pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=3, strides=3, padding='same')

        conv3 = tf.layers.conv2d(inputs = max_pool2, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        max_pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=3, strides=3, padding='same')

        conv4 = tf.layers.conv2d(inputs = max_pool3, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        max_pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=3, strides=3, padding='same')

        flat = tf.layers.flatten(inputs=max_pool4)

        logits = tf.layers.dense(inputs=flat, units=1)
        logits = tf.squeeze(logits)

        predictions = tf.round(tf.nn.sigmoid(logits))
        
        loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=logits,
                labels=y_))
        tf.summary.scalar('loss', loss)
        
        train = tf.train.AdamOptimizer(1e-4).minimize(loss)
        
        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, y_), dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        frac = tf.reduce_mean(y_)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter('/tmp/tf/cnn', graph)
        print(f"Logging to /tmp/tf/cnn")

        saver = tf.train.Saver(tf.trainable_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            if False:
                saver.restore(sess, '../models/cnn.ckpt')
            
            try:
                for i in range(10001):
                    images, labels = data.next()
                    print(images.shape)
                    sess.run(train, feed_dict={x:images, y_:labels, training: True})

                    if i % 10 == 0:
                        train_acc = sess.run(accuracy, feed_dict={x:images, y_:labels, training: False})
                        images, labels = data.validation()
                        summary, acc, frac_ = sess.run([merged, accuracy, frac], feed_dict={x:images, y_:labels, training: False})
                        writer.add_summary(summary, i)
                        print(f"{i}: train acc: {train_acc:6.4f} acc {acc:6.4f} with frac: {frac_}")

                    if i % 20 == 0 and i != 0:
                        saver.save(sess, '../models/cnn.ckpt', global_step=i)

            except KeyboardInterrupt:
                print(f"Training interrupted. Now saving model at step {i}.")
            finally:
                saver.save(sess, '../models/cnn.ckpt', global_step=i)

if __name__ == '__main__':
    model(True)
