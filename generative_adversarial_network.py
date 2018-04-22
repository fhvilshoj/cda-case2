import tensorflow as tf
import numpy as np
import skimage.io as io
import argparse
import os
import os.path

from tfrecord_io import read_and_decode


def setup_input(config):
    images, labels = read_and_decode(config.tf_records, batch_size=config.batch_size)
    return images, labels

def main(config):
    with tf.Graph().as_default() as graph:
        
        images, _ = setup_input(config)

        # TODO Build discriminator and generator
        
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        with tf.Session() as sess:
            sess.run(init_op)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            for i in range(2):
                img, ls = sess.run([images, labels])

                io.imshow(img[0])
                io.show()
        
            coord.request_stop()
            coord.join(threads)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='generative_adversarial_network.py',
                                     description='GAN network to pretrain weights for haze detection',
                                     epilog='This is cool stuff')

    parser.add_argument('-b', '--batch-size', type=int, default=50,
                        help='Batch size for training')
    parser.add_argument('-tfr', '--tf-records', nargs='+',
                        help='Records to load images from')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='Restore learned parameters')
    parser.add_argument('-m', '--model', type=str, default='../models/gan.ckpt',
                        help='Model to store and restore parameters from')
    
    args = parser.parse_args()
    
    main(args)
