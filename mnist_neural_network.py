import os
import numpy as np
import tensorflow as tf

from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets('/tmp/data', one_hot=True)

inputs = 28**2
hl_nodes = [400, 400, 400]

num_classes = 10
batch_size = 100

# height & width
x = tf.placeholder('float', [None, inputs])
y = tf.placeholder('float')

def neural_network_model_old(data):
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

def neural_network_model(data):
    hl_weights = []
    a_values = []
    
    for i, hl_n in enumerate(hl_nodes):
        dim1 = hl_nodes[i - 1] if i > 0 else inputs
        dim2 = hl_n

        hl_weights.append({
            'weights': tf.Variable(tf.random_normal([dim1, dim2])),
            'biases': tf.Variable(tf.random_normal([dim2]))
        })

        x = a_values[i - 1] if i > 0 else data

        a = tf.matmul(x, hl_weights[i]['weights']) + hl_weights[i]['biases']
        a_values.append(tf.nn.relu(a))

    out_weights = {'weights': tf.Variable(tf.random_normal([hl_nodes[-1], num_classes])),
                     'biases': tf.Variable(tf.random_normal([num_classes]))}

    output = tf.matmul(a_values[-1], out_weights['weights']) + out_weights['biases']

    return output

def train_neural_network(x, y):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

    optimizer = tf.train.AdamOptimizer().minimize(cost)
    hm_epochs = 10

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

                epoch_loss += c

            print('Epoch ', epoch + 1, ' completed out of ', hm_epochs, '; Loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))

        saver.save(sess, os.path.join(os.getcwd(), 'checkpoints/mnist_model.ckpt'))

    return saver

def predict(input_data):
    prediction = neural_network_model(x)
    saver = tf.train.Saver()
    result = None

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, tf.train.latest_checkpoint(os.path.join(os.getcwd(), 'checkpoints/')))
        result = sess.run(tf.argmax(prediction.eval(feed_dict={x: input_data}), axis=1))

    return result

def convert_to_mnist(path):
    img = Image.open(path).convert('L').resize([28, 28])
    img_arr = np.array(img).flatten()
    return 1 - img_arr/max(img_arr)

# Train neural network
# train_neural_network(x, y)

# Predict sample mnist and own images
input_data = np.array([mnist.train.images[0],
                      mnist.train.images[1],
                      convert_to_mnist('digit_images/Three.jpg'),
                      convert_to_mnist('digit_images/Five.jpg')])

results = predict(input_data)
print('Prediction: ', results[0], 'Actual:', np.argmax(mnist.train.labels[0]))
print('Prediction: ', results[1], 'Actual:', np.argmax(mnist.train.labels[1]))
print('Prediction: ', results[2], 'Actual: 3')
print('Prediction: ', results[3], 'Actual: 5')

