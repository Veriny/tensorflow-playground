import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#Datasets
mnist = input_data.read_data_sets("/tmp/data/", one_hot = True)

#AI Knobs and Bobs
learning_rate = 0.01
batch_size = 10
label_rate_thing = 2
training_iteration = 30

x = tf.placeholder("float", [None, 784]) #MNIST dataset images are 28 * 28 = 784 pixels in area, so we take in a tensor of 784 pixels in area
y = tf.placeholder("float", [None, 10]) #There are 10 possibilities for numbers, so we will have 10 possible y values.
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

with tf.name_scope("Wx_b") as scppe:
    model = tf.nn.softmax(tf.matmul(x, W) + b)

#w_h = tf.histogram_summary("weights", W)
#b_h = tf.histogram_summary("biases", b)

with tf.name_scope("cost_function") as scope:
    cost_function = -tf.reduce_sum(y*tf.log(model))

with tf.name_scope("train") as scope:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

init = tf.global_variables_initializer()
merge_summary_op = tf.summary.merge_all()

#training time my ddues
with tf.Session() as sess:
    sess.run(init)
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        for batch in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            result = sess.run(optimizer, feed_dict = {x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost_function, feed_dict = {x: batch_xs, y: batch_ys})/total_batch

print(avg_cost)
