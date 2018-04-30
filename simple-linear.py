import tensorflow as tf
import numpy as np
import skimage.io as io
import argparse
import random
import string

from tfrecord_io import read_and_decode


records = ['../data/tfrecords/test3-1.tfrecords', '../data/tfrecords/test3-2.tfrecords']

def main(config):
    with tf.Graph().as_default() as graph:
        
        images, labels = read_and_decode(config.tf_records, batch_size=config.batch_size)
        batch_size, height, width, channels = images.shape.as_list()
        
        images = tf.reshape(images, (-1, height * width * channels))
        
        dense1 = tf.layers.dense(
            inputs=images,
            units=500,
            activation=tf.nn.relu)

        dense2 = tf.layers.dense(
            inputs=dense1,
            units=100,
            activation=tf.nn.relu)

        logits = tf.layers.dense(
            inputs=dense2,
            units=1)
        logits = tf.squeeze(logits)
        print(logits)

        predictions = tf.round(tf.nn.sigmoid(logits))
        
        loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits,
                    labels=tf.cast(labels, dtype=tf.float32)))
        tf.summary.scalar('loss', loss)
        
        train = tf.train.AdamOptimizer().minimize(loss)

        accuracy = tf.reduce_mean(tf.cast(tf.equal(predictions, tf.cast(labels, dtype=tf.float32)), dtype=tf.float32))
        tf.summary.scalar('accuracy', accuracy)

        frac = tf.reduce_mean(tf.cast(labels, tf.float32))

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.logdir, graph)
        print(f"Logging to {config.logdir}")

        saver = tf.train.Saver(tf.trainable_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            if config.restore:
                saver.restore(sess, config.model)
            
            try:
                for i in range(10001):
                    sess.run(train)

                    if i % 10 == 0:
                        summary, acc, loss_, frac_ = sess.run([merged, accuracy, loss, frac])
                        writer.add_summary(summary, i)
                        print(f"{i}: acc {acc:6.4f} loss: {loss_:6.4f} with frac: {frac_}")

                    if i % 1000 == 0 and i != 0:
                        saver.save(sess, config.model, global_step=i)
                        
            except KeyboardInterrupt:
                print(f"Training interrupted. Now saving model at step {i}.")
            finally:
                saver.save(sess, config.model, global_step=i)
                coord.request_stop()
                coord.join(threads)

def id_generator(size=6, chars=string.ascii_letters + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog="Simple 3 layer perceptron")

    parser.add_argument('-b', '--batch-size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('-tfr', '--tf-records', nargs='+',
                        help='Records to load images from')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='Restore learned parameters')
    parser.add_argument('-m', '--model', type=str, default='../models/simple-mlp.ckpt',
                        help='Model to store and restore parameters from')
    parser.add_argument('--logdir', type=str, default=f'/tmp/tf/mlp/{id_generator()}',
                        help='Tensorboard summary logdir')
    
    args = parser.parse_args()
    main(args)
