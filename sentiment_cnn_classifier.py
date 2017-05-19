import nltk
import math
import numpy as np
import pickle
import random
import tensorflow as tf

from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split


class SentimentCNNClassifier():

    def __init__(self, lb=30, ub=1000, a_func=tf.nn.relu, test_size=0.2):
        self.lemmatizer = WordNetLemmatizer()
        self.ub = ub
        self.lb = lb
        self.test_size = test_size
        self.a_func = a_func
        
    def create_lexicon(self, text_docs):
        all_words = []
        for doc in text_docs:
            all_words += word_tokenize(doc)
            
        lemmatized_words = [self.lemmatizer.lemmatize(w.lower()) for w in all_words]
        word_dist = nltk.FreqDist(lemmatized_words)
        lexicon = set(lemmatized_words)

        # Remove common words
        lexicon = [w for w in lexicon if self.lb < word_dist[w] < self.ub]

        return lexicon

    def process_features_and_labels(self, documents):
        # Return from pickle file if one exists
        try:
            with open('pickled_files/docs_x_y.pickle', 'rb') as file:
                [X, y] = pickle.load(file)

            return X, y
        except FileNotFoundError:
            pass
        
        # Create lexicon
        text_samples = [d[0] for d in documents]
        lexicon = self.create_lexicon(text_samples)

        # Create feature set
        X = []
        y = []
        # feature_set = []
        for doc in documents:
            features = np.zeros(len(lexicon))

            words = word_tokenize(doc[0])
            lemmatized_words = [self.lemmatizer.lemmatize(w.lower()) for w in words]

            for w in lemmatized_words:
                if w in lexicon:
                    features[lexicon.index(w)] += 1

            # feature_set.append([features, doc[1]])
            X.append(features)
            y.append(doc[1])

        # Write X and y variables to pickle
        with open('pickled_files/docs_x_y.pickle', 'wb') as file:
            pickle.dump([X, y], file)

        return X, y

    def neural_network_model(self, data, input_nodes, hl_nodes, num_classes):
        hl_weights = []
        a_values = []
        for i, hl_n in enumerate(hl_nodes):
            dim1 = hl_nodes[i - 1] if i > 0 else input_nodes
            dim2 = hl_n

            hl_weights.append({
                'weights': tf.Variable(tf.random_normal([dim1, dim2])),
                'biases': tf.Variable(tf.random_normal([dim2]))
            })

            x = a_values[i - 1] if i > 0 else data

            a = tf.matmul(x, hl_weights[i]['weights']) + hl_weights[i]['biases']
            a_values.append(self.a_func(a))

        out_weights = {'weights': tf.Variable(tf.random_normal([hl_nodes[-1], num_classes])),
                         'biases': tf.Variable(tf.random_normal([num_classes]))}

        output = tf.matmul(a_values[-1], out_weights['weights']) + out_weights['biases']

        self.nn_model = output
        return output

    def train_neural_network(self, documents, hl_nodes, num_classes, hm_epochs=10, batch_size=100):
        # Process and split data
        X, y = self.process_features_and_labels(documents)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size)
        print('Data processing complete')

        # Set x and y placeholders
        input_nodes = len(X_train[0])
        x = tf.placeholder('float', [None, input_nodes])
        y = tf.placeholder('float')

        # Set cost function
        prediction = self.neural_network_model(x, input_nodes, hl_nodes, num_classes)
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Run tensorflow session
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for epoch in range(hm_epochs):
                epoch_loss = 0

                for i in range(math.ceil(len(X_train)/batch_size)):
                    ub = min(i*batch_size + batch_size - 1, len(documents) - 1)
                    X_epoch = X_train[i*batch_size:ub]
                    y_epoch = y_train[i*batch_size:ub]
    
                    _, c = sess.run([optimizer, cost], feed_dict={x: X_epoch, y: y_epoch})
                    epoch_loss += c

                print('Epoch ', epoch, ' completed out of ', hm_epochs, '; Loss: ', epoch_loss)

            correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
            print('Accuracy: ', accuracy.eval({x: X_test, y: y_test}))

        
# Neural network model parameters
hl_nodes = [300, 300, 300]
num_classes = 2

pos_file = open('short_reviews/positive.txt', 'r', encoding='latin-1').read()
neg_file = open('short_reviews/negative.txt', 'r', encoding='latin-1').read()
reviews = [(r, [1, 0]) for r in pos_file.split('\n')] + [(r, [0, 1]) for r in neg_file.split('\n')]

classifier = SentimentCNNClassifier(test_size=0.2)
classifier.train_neural_network(reviews, hl_nodes, num_classes, batch_size=100)


