import tensorflow as tf
import skimage.io as io

IMAGE_HEIGHT = 272
IMAGE_WIDTH = 384

tfrecord = '../data/tfrecords/test3-1.tfrecords'

def read_and_decode(input_files, num_epochs=10, batch_size=5):
    filename_queue = tf.train.string_input_producer(
        input_files, num_epochs=num_epochs)
    
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'label': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })

    image = tf.decode_raw(features['image_raw'], tf.uint8)

    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    label = tf.cast(features['label'], tf.int32)

    image_shape = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = tf.reshape(image, image_shape)

    images, labels = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        capacity=30,
        num_threads=2,
        min_after_dequeue=10)
    
    return images, labels

