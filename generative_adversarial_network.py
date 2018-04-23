import tensorflow as tf
import numpy as np
import skimage.io as io
import argparse
import os
import os.path
import random
import string

from tfrecord_io import read_and_decode
from generator import generator
from discriminator import discriminator

def setup_input(config):
    images, labels = read_and_decode(config.tf_records, batch_size=config.batch_size)
    return images, labels

def main(config):
    with tf.Graph().as_default() as graph:
        is_training = tf.placeholder(dtype=tf.bool)
        
        with tf.name_scope("load_images"):
            images, _ = setup_input(config)
            tf.summary.image('input', images[:,:,:,:3], 1)
            
        with tf.name_scope("rangom_noise"):
            zs = tf.random_normal(shape=(config.batch_size, 100))

        channels = images.shape.as_list()[-1]
        
        with tf.name_scope("generator"):
            fakes = generator(zs, channels, is_training)
            tf.summary.image('fakes', fakes[:,:,:,:3], 1)
            
        with tf.name_scope("discriminator"):
            Dimages = discriminator(images, channels, is_training)
            Dfakes = discriminator(fakes, channels, is_training, reuse=True)
            

        # Generator loss
        with tf.name_scope("generator_loss"):
            g_loss = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = Dfakes,
                    labels = tf.ones_like(Dfakes)))
            tf.summary.scalar('g_loss', g_loss)

        # Discriminator loss
        with tf.name_scope("discriminator_loss"):
            d_loss_real = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = Dimages,
                    labels = tf.ones_like(Dimages)))
            tf.summary.scalar('d_loss_real', d_loss_real)
            d_loss_fake = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    logits = Dfakes,
                    labels = tf.zeros_like(Dfakes)))
            tf.summary.scalar('d_loss_fake', d_loss_fake)
            d_loss = d_loss_real + d_loss_fake
            tf.summary.scalar('d_loss', d_loss)
        

        # Separate discriminator and generator variables 
        tvars = tf.trainable_variables()
        dis_vars = [var for var in tvars if 'dis_' in var.name]
        gen_vars = [var for var in tvars if 'gen_' in var.name]

        with tf.variable_scope(tf.get_variable_scope(),reuse=False): 
            print("reuse or not: {}".format(tf.get_variable_scope().reuse))
            assert tf.get_variable_scope().reuse == False, "Houston we have a problem"
            trainerD = tf.train.AdamOptimizer().minimize(d_loss, var_list=dis_vars)
            trainerG = tf.train.AdamOptimizer().minimize(g_loss, var_list=gen_vars)

        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(config.logdir, graph)
        print(f"Logging to {config.logdir}")
                
        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())

        saver = tf.train.Saver(tvars)

        with tf.Session() as sess:
            sess.run(init_op)

            if config.restore:
                saver.restore(sess, config.model)
            
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)

            try:
                for i in range(3001):
                    _, dLoss = sess.run([trainerD, d_loss], {is_training: True})
                    _, gLoss = sess.run([trainerG, g_loss], {is_training: True})

                    if i % 10 == 0:
                        dLoss, gLoss, summ = sess.run([d_loss, g_loss, merged], {is_training: False})
                        writer.add_summary(summ, i)
                        print(f"{i}: dl {dLoss:6.4f} gl: {gLoss:6.4f}")

                    if i != 0 and i % 1000 == 0:
                        print(f"Saving model at step {i}")
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
    parser = argparse.ArgumentParser(prog='generative_adversarial_network.py',
                                     description='GAN network to pretrain weights for haze detection',
                                     epilog='This is cool stuff')

    parser.add_argument('-b', '--batch-size', type=int, default=20,
                        help='Batch size for training')
    parser.add_argument('-tfr', '--tf-records', nargs='+',
                        help='Records to load images from')
    parser.add_argument('-r', '--restore', action='store_true',
                        help='Restore learned parameters')
    parser.add_argument('-m', '--model', type=str, default='../models/gan.ckpt',
                        help='Model to store and restore parameters from')
    parser.add_argument('--logdir', type=str, default=f'/tmp/tf/gan/{id_generator()}',
                        help='Tensorboard summary logdir')
    
    args = parser.parse_args()
    
    main(args)
