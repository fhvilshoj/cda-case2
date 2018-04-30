import tensorflow as tf
import skimage.io as io
import re

IMAGE_HEIGHT = 272
IMAGE_WIDTH = 384

tfrecord = '../data/tfrecords/test3-1.tfrecords'

def read_and_decode(input_files, num_epochs=10, batch_size=5):
    print(input_files)
    filename_queue = tf.train.string_input_producer(
        input_files, num_epochs=num_epochs)
    print(filename_queue)
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image = (tf.cast(image, dtype=tf.float32)-127.5)/127.5
    label = tf.cast(features['label'], tf.int32)

    pattern = r'.*_d(?P<depth>\d+)'
    image_depth = int(re.match(pattern, input_files[0]).group('depth'))
    image_shape = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, image_depth])
    
    image = tf.reshape(image, image_shape)

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=200,
        num_threads=1,
        min_after_dequeue=150)
    
    return images, labels

