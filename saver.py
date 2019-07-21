import tensorflow as tf
import numpy as np


# W = tf.Variable([[1,2,3],[3,4,5]],dtype=tf.float32,name='weifhts')
# b = tf.Variable([[1,2,3]],dtype=tf.float32,name='biases')
#
# init = tf.initialize_all_variables()
#
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#     save_path = saver.save(sess,"my_net/save_net.ckpt")
#     print("Save tp path",save_path)

#restore variables
#redifine the same shape and same type

tf.reset_default_graph()
W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name="weights")
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name="biases")

saver = tf.train.Saver()
module_file = tf.train.latest_checkpoint('my_net/save_net.ckpt')
with tf.Session() as sess:
   if module_file is not None:
      saver.restore(sess, module_file)
    print("weights:",sess.run(W))
    print("biases:", sess.run(b))