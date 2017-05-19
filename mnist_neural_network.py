import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

inputs = 28**2
hl1_nodes = 400
hl2_nodes = 400
hl3_nodes = 400

num_classes = 10
batch_size = 100

# height & width
x = tf.placeholder('float', [None, inputs])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([inputs, hl1_nodes])),
                     'biases': tf.Variable(tf.random_normal([hl1_nodes]))}

    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([hl1_nodes, hl2_nodes])),
                     'biases': tf.Variable(tf.random_normal([hl2_nodes]))}

    hidden_layer3 = {'weights': tf.Variable(tf.random_normal([hl2_nodes, hl3_nodes])),
                     'biases': tf.Variable(tf.random_normal([hl3_nodes]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([hl3_nodes, num_classes])),
                     'biases': tf.Variable(tf.random_normal([num_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_layer2['weights']), hidden_layer2['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_layer3['weights']), hidden_layer3['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

    return output

def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch ', epoch, ' completed out of ', hm_epochs, '; Loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


train_neural_network(x, y)
