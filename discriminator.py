import tensorflow as tf
from constants import IMAGE_HEIGHT, IMAGE_WIDTH

def discriminator(x, channels, is_training, reuse=False):
    with tf.name_scope("dis_reshape"):
        # Shape (None, 272, 384, channels)
        x_input = tf.reshape(x, (-1, IMAGE_HEIGHT, IMAGE_WIDTH, channels))

    with tf.name_scope("dis_conv1"):
        conv1 = tf.layers.conv2d(
            inputs=x_input,
            filters=32,
            kernel_size=[5,5],
            padding='same',
            name="dis_conv1",
            reuse=reuse)
        batch1 = tf.layers.batch_normalization(
            inputs=conv1,
            renorm=True,
            training=is_training,
            name="dis_bn1",
            reuse=reuse)
        relu1 = tf.nn.relu(batch1)
        max1 = tf.layers.max_pooling2d(
            inputs=relu1,
            pool_size=[3, 3],
            strides=3)
        # output (None, 90, 128, 32)
        
    with tf.name_scope("dis_conv2"):
        conv2 = tf.layers.conv2d(
            inputs=max1,
            filters=48,
            kernel_size=[4,4],
            padding='same',
            name="dis_conv2",
            reuse=reuse)
        batch2 = tf.layers.batch_normalization(
            inputs=conv2,
            renorm=True,
            training=is_training,
            name="dis_bn2",
            reuse=reuse)
        relu2 = tf.nn.relu(batch2)
        max2 = tf.layers.max_pooling2d(
            inputs=relu2,
            pool_size=[2, 2],
            strides=2)
        # output (None, 45, 64, 48)

    with tf.name_scope("dis_conv3"):
        conv3 = tf.layers.conv2d(
            inputs=max2,
            filters=64,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu,
            name="dis_conv3",
            reuse=reuse)
        max3 = tf.layers.max_pooling2d(
            inputs=conv3,
            pool_size=[2,2],
            strides=2)
        # output (None, 22, 32, 64)
    
    with tf.name_scope("dis_conv4"):
        conv4 = tf.layers.conv2d(
            inputs=max3,
            filters=64,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu,
            name="dis_conv4",
            reuse=reuse)
        max4 = tf.layers.max_pooling2d(
            inputs=conv4,
            pool_size=[2,2],
            strides=2)
        # output (None, 11, 16, 64)

    with tf.name_scope("dis_conv5"):
        conv5 = tf.layers.conv2d(
            inputs=max4,
            filters=128,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu,
            name="dis_conv5",
            reuse=reuse)
        max5 = tf.layers.max_pooling2d(
            inputs=conv5,
            pool_size=[2,2],
            strides=2)
        # output (None, 5, 8, 128)
    
    with tf.name_scope("dis_conv6"):
        conv6 = tf.layers.conv2d(
            inputs=max5,
            filters=128,
            kernel_size=[3,3],
            padding='same',
            activation=tf.nn.relu,
            name="dis_conv6",
            reuse=reuse,
        )
        max6 = tf.layers.max_pooling2d(
            inputs=conv6,
            pool_size=[2,2],
            strides=2)
        # output (None, 2, 4, 128)

    with tf.name_scope("dis_flatten"):
        flat = tf.layers.flatten(inputs=max6)
        # output (None, 1024)

    with tf.name_scope("dis_dense1"):
        dense1 = tf.layers.dense(
            inputs=flat,
            units=200,
            activation=tf.nn.relu,
            name="dis_dense1",
            reuse=reuse)

    with tf.name_scope("dis_dropout"):
        drop = tf.layers.dropout(
            inputs=dense1,
            rate=0.9,
            training=is_training)

    with tf.name_scope("dis_dense2"):
        dense2 = tf.layers.dense(
            inputs=drop,
            units=2,
            name="dis_dense2",
            reuse=reuse)

    return dense2
