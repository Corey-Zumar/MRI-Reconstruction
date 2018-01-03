import tensorflow as tf
import numpy as np

def create_test_arch():
	sess = tf.Session()

	with sess.graph.as_default():
		t_inputs = tf.placeholder(tf.float32, shape=[None, 2, 2, 1])

		t_avgpool = tf.nn.max_pool(t_inputs, ksize=(1,2,2,1), strides=(1,1,1,1), padding="VALID")

		t_grads = tf.gradients(t_avgpool, t_inputs)

	return sess, t_inputs, t_avgpool, t_grads

if __name__ == "__main__":
	sess, t_inputs, t_avgpool, t_grads = create_test_arch()

	input_items = np.random.rand(2,2,1) * 10

	feed_dict = {
		t_inputs : np.array([input_items])
	}

	print(input_items)

	avgs, grads = sess.run([t_avgpool, t_grads], feed_dict=feed_dict)

	print(avgs)
	print(grads)
