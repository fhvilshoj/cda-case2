import tensorflow as tf
import numpy as np
import skimage.io as io

from tfrecord_io import read_and_decode


records = ['../data/tfrecords/test3-1.tfrecords', '../data/tfrecords/test3-2.tfrecords']

images, labels = read_and_decode(records)

init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(2):
        img, ls = sess.run([images, labels])

        print(ls)
        
        io.imshow(img[0])
        io.show()
    
    coord.request_stop()
    coord.join(threads)
