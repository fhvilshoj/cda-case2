import numpy as np
import tensorflow as tf

def _next_batch():
    return [], []

def _test():
    return [], []

def model(self, is_training):

    with tf.Graph().as_default() as graph:

        x = tf.placeholder(dtype=tf.float32, shape=(None, 170, 384))
        y_ = tf.placeholder(dtype=tf.float32, shape=(None,))
            
        training = tf.placeholder(dtype=tf.bool)
            
        x = tf.reshape(x, (None, 170, 384, 1))

        conv1 = tf.layers.conv2d(inputs = x, filters=8, kernel_size=5, padding='same')
        batch1 = tf.layers.batch_normalization(inputs=conv1, is_training=training)
        relu1 = tf.nn.reul(batch1)
        max_pool1 = tf.layers.max_pooling2d(inputs=relu1, pool_size=3, strides=3, padding='same')
        
        conv2 = tf.layers.conv2d(inputs = max_pool1, filters=16, kernel_size=5, padding='same')
        batch2 = tf.layers.batch_normalization(inputs=conv2, is_training=training)
        relu2 = tf.nn.reul(batch2)
        max_pool2 = tf.layers.max_pooling2d(inputs=relu2, pool_size=3, strides=3, padding='same')

        conv3 = tf.layers.conv2d(inputs = max_pool2, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        max_pool3 = tf.layers.max_pooling2d(inputs=conv3, pool_size=3, strides=3, padding='same')

        conv4 = tf.layers.conv2d(inputs = max_pool3, filters=32, kernel_size=5, padding='same', activation=tf.nn.relu)
        max_pool4 = tf.layers.max_pooling2d(inputs=conv4, pool_size=3, strides=3, padding='same')

        flat = tf.layers.flatten(inputs=max_pool4)

        logits = tf.layers.dense(inputs=flat, units=2)
        logits = tf.squeeze(logits)

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
        writer = tf.summary.FileWriter('/tmp/tf/cnn', graph)
        print(f"Logging to /tmp/tf/cnn")

        saver = tf.train.Saver(tf.trainable_variables())

        init_op = tf.group(tf.global_variables_initializer(),
                           tf.local_variables_initializer())
        
        
        with tf.Session() as sess:
            sess.run(init_op)
            
            if False:
                saver.restore(sess, config.model)
            
            try:
                for i in range(10001):
                    images, labels = _next_batch()
                    sess.run(train, feed_dict={x:images, y_:labels, is_training: True})

                    if i % 10 == 0:
                        images, labels = 
                        summary, acc, loss_, frac_ = sess.run([merged, accuracy, loss, frac], , feed_dict={x:images, y_:labels, is_training: True})
                        writer.add_summary(summary, i)
                        print(f"{i}: acc {acc:6.4f} loss: {loss_:6.4f} with frac: {frac_}")

                    if i % 100 == 0 and i != 0:
                        saver.save(sess, '../../models/cnn.ckpt', global_step=i)
                    
            except KeyboardInterrupt:
                print(f"Training interrupted. Now saving model at step {i}.")
            finally:
                saver.save(sess, config.model, global_step=i)
                
            
