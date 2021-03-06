
import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim


"""
I need to test in a different place that targets always has a 1 in it. That's just for the accuracy claim though, so it's not a huge problem
"""

def accuracy_function(outputs, targets):
    # How's it gonna work? It's going to ask, is the target among the biggest values in the target array? I think all I need to ask is, is it 1?
    # 
    assert outputs.shape == targets.shape
    assert len(outputs.shape) == 2

    # number_of_ones = 0
    # for val in np.nditer(targets):
    #     if val == 1:
    #         number_of_ones += 1
    # number_of_elements = targets.shape[0]*targets.shape[1]
    # print("Frequency of ones: {}".format(number_of_ones/number_of_elements))

    indices = np.argmax(outputs, axis=1) #these should be the maximum values of each one.
    extracted_vals = [t[i] for i, t in zip(indices, targets)]
    true_vals = np.asarray([1.0 if val == 1.0 else 0 for val in extracted_vals])
    average_accuracy = np.mean(true_vals)
    # print(average_accuracy)
    return average_accuracy

class SpiderLocator(object):
    # Takes in something like 25 inputs. Outputs something like 4 values, between -1 and 1.
    # 1 means closer, -1 means farther. You choose the one that seems like it'll get you most closest.

    def __init__(self,
                 samples=None,
                 targets=None,
                 mean_samples=0.0,
                 variance_samples=1.0):
        assert None not in [samples, targets]

        self.samples = samples
        self.targets = targets
        self.mean_samples = mean_samples
        self.variance_samples=variance_samples
        
        self.initialize()
    
    def initialize(self):
        self.construct_forward_pass()
        self.construct_training()
        self.construct_logging()
        self.saver = tf.train.Saver()

    def construct_forward_pass(self):
        # h = self.samples
        # h = slim.fully_connected(h, 50, activation_fn=tf.nn.elu)
        # h = slim.fully_connected(h, 25, activation_fn=tf.nn.elu)
        # h = slim.fully_connected(h, 4, activation_fn=tf.nn.tanh)
        # self.network_output = h
        h = self.samples
        h = slim.fully_connected(h, 1000, activation_fn=tf.nn.elu)
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.elu)
        h = slim.fully_connected(h, 4, activation_fn=tf.nn.tanh)
        self.network_output = h

        self.prediction = tf.argmax(self.network_output, axis=1) # I think that's right.

        pass
    
    def construct_training(self):
        squared_difference = tf.squared_difference(self.targets, self.network_output)
        mean_square_error = tf.reduce_mean(squared_difference)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(mean_square_error)
        # train_op = tf.train.GradientDescentOptimizer(1e-4).minimize(mean_square_error)

        self.error = mean_square_error
        self.train_op = train_op
        self.accuracy = tf.py_func(accuracy_function, [self.network_output, self.targets], tf.float64)

    def construct_logging(self):
        training_error = tf.summary.scalar('training/mse', self.error)
        training_accuracy = tf.summary.scalar('training/accuracy', self.accuracy)
        testing_error = tf.summary.scalar('testing/mse', self.error)
        testing_accuracy = tf.summary.scalar('testing/accuracy', self.accuracy)

        training_summaries = tf.summary.merge([training_error, training_accuracy])

        testing_summaries = tf.summary.merge([testing_error, testing_accuracy])

        self.training_summaries = training_summaries
        self.testing_summaries = testing_summaries

    pass
