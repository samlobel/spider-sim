
import tensorflow as tf
import numpy as np

import tensorflow.contrib.slim as slim


"""
I need to test in a different place that targets always has a 1 in it. That's just for the accuracy claim though, so it's not a huge problem
"""

# def accuracy_function(outputs, targets):
#     assert outputs.shape == targets.shape
#     assert len(outputs.shape) == 2
#     indices = np.argmax(outputs, axis=1) #these should be the maximum values of each one.
#     extracted_vals = [t[i] for i, t in zip(indices, targets)]
#     true_vals = np.asarray([for val in extracted_vals])

#     max_in_each_target_row = np.amax(targets, axis=1)
#     all_ones = np.ones_like(max_in_each_target_row)
#     values_at_indices = 

class SpiderModel(object):
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

    def construct_forward_pass(self):
        h = self.samples
        h = slim.fully_connected(h, 100, activation_fn=tf.nn.elu)
        h = slim.fully_connected(h, 10, activation_fn=tf.nn.elu)
        h = slim.fully_connected(h, 4, activation_fn=tf.nn.tanh)
        self.network_output = h

        self.prediction = tf.argmax(self.network_output, axis=1) # I think that's right.

        pass
    
    def construct_training(self):
        squared_difference = tf.squared_difference(self.targets, self.network_output)
        mean_square_error = tf.reduce_mean(squared_difference)
        train_op = tf.train.AdamOptimizer(1e-4).minimize(mean_square_error)

        self.error = mean_square_error
        self.train_op = train_op

    def predict(self):


    # def compute_accuracy(self):



    def construct_logging(self):
        training_error = tf.summary.scalar('training/ce_loss', self.loss)

        testing_error = tf.summary.scalar('training/ce_loss', self.loss)

    pass
