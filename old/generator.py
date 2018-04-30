import tensorflow as tf

from constants import IMAGE_HEIGHT, IMAGE_WIDTH

# z of shape 100
def generator(z, channels, is_training):
    #         272           384
    h, w = IMAGE_HEIGHT, IMAGE_WIDTH

    h2, h4, h8, h16 = int(h/2), int(h/4), int(h/8), int(h/16)
    w2, w4, w8, w16 = int(w/2), int(w/4), int(w/8), int(w/16) 
    
    with tf.name_scope("gen_project"):
        z_scaled = tf.layers.dense(
            inputs=z,
            units=17*24*64,
            activation=tf.nn.relu,
            name="gen_dense")
    
    with tf.name_scope("gen_reshape"):
        z_input = tf.reshape(z_scaled, (-1, 17, 24, 64))

    with tf.name_scope("gen_deconv1"):
        deconv1 = tf.layers.conv2d_transpose(
            inputs=z_input,
            filters=32,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            name="gen_deconv1")
        bn1 = tf.layers.batch_normalization(
            inputs=deconv1,
            renorm=True,
            training=is_training,
            name="gen_bn1")
        relu1 = tf.nn.relu(bn1)
        # output (None, 34, 48, 32)

    with tf.name_scope("gen_deconv2"):
        deconv2 = tf.layers.conv2d_transpose(
            inputs=relu1,
            filters=16,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            name="gen_deconv2")
        bn2 = tf.layers.batch_normalization(
            inputs=deconv2,
            renorm=True,
            training=is_training,
            name="gen_bn2")
        relu2 = tf.nn.relu(bn2)
        # output (None, 68, 96, 16)
        
    with tf.name_scope("gen_deconv3"):
        deconv3 = tf.layers.conv2d_transpose(
            inputs=relu2,
            filters=8,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=tf.nn.relu,
            name="gen_deconv3")
        # output (None, 136, 192, 8)
        
    with tf.name_scope("gen_deconv4"):
        deconv4 = tf.layers.conv2d_transpose(
            inputs=deconv3,
            filters=channels,
            kernel_size=[5,5],
            strides=[2,2],
            padding='same',
            activation=tf.nn.tanh,
            name="gen_deconv4")
        # output (None, 272, 384, channels)

    return deconv4
        
            
    
