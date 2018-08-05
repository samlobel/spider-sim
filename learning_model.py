
import tensorflow as tf
import numpy as np

from tensorflow import layers


class SpiderModel(object):
    """
    What are the integral parts? It needs to know the dimensions of its input space.
    For now it's just classification, so its loss function is CE.

    For testing error, I'll use a placeholder. That's an easy way of doing it.

    I'd like it to be batch-independent.

    Side Note: I feel like I messed up with the data format.
    I should have done it as [batch_size, num_samples, num_channels].
    That just makes more logical sense. But, that's a problem for another day.

    I NEED TO SET DATA_FORMAT FOR CONV1D.


    Question: Should I use tf.one_hot? It is pretty nice.
    For now, I will.
    """

    def __init__(
        self,
        num_channels=24,
        num_timesteps=500,
        scale_conv_regularizer=0.0,
        *args,
        **kwargs):
        self.num_channels = num_channels
        self.num_timesteps = num_timesteps
        self.scale_conv_regularizer = scale_conv_regularizer

        self.create_placeholders()
        self.create_prediction_model()

    def create_placeholders(self):
        self.input_ph = tf.placeholder(
            dtype=tf.float32, shape=[None, self.num_channels, self.num_timesteps])
        self.target_ph = tf.placeholder(dtype=tf.float32, shape=[None, 4])

        self.testing_loss = tf.placeholder(dtype=tf.float32, shape=[])
        self.testing_accuracy  = tf.placeholder(dtype=tf.float32, shape=[])
        pass

    def create_outputs(self):
        one_hot_out = tf.one_hot(self.target_ph)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_out, logits=self.logits_out)
        output_probabilities = tf.nn.softmax(self.logits_out)

        # correct_prediction = tf.equal(tf.argmax(self.logits_out, 1), tf.argmax(self.target_ph, 1))
        correct_prediction = tf.equal(tf.argmax(self.logits_out, 1), self.target_ph)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        self.loss = loss
        self.output_probabilities = output_probabilities
        self.accuracy = accuracy


    def create_prediction_model(self):
        return self._simple_create_prediction_model()


    def create_logging(self):
        """
        Construct logging for:
        Training error.
        Classification percentage.

        Testing Error/Testing Classification Percentage (off of placeholders).
        """

        tr_ce = tf.summary.scalar('training/ce_loss', self.loss)
        tr_acc = tf.summary.scalar('testing/accuracy', self.accuracy)

        te_ce = tf.summary.scalar('testing/ce_loss', self.testing_loss)
        te_acc = tf.summary.scalar('testing/accuracy', self.testing_accuracy)

        training_summaries = tf.summary.merge([tr_ce, tr_acc])
        testing_summaries = tf.summary.merge([te_ce, te_acc])

        self.training_summaries = training_summaries
        self.testing_summaries = testing_summaries

    def _simple_create_prediction_model(self):
        # This one is going to be a lot simpler. It's going to use the intensity
        # of each "data row", and have num_channels inputs. That's way better to start.

        input_intensities = self.input_ph * self.input_ph
        average_over_time_intensities = tf.reduce_mean(input_intensities, axis=-1)
        shape = average_over_time_intensities.get_shape()
        assert len(shape) == 2
        assert shape[1] == self.num_channels

        out = average_over_time_intensities
        out = \
            tf.layers.dense(
                out,
                100,
                activation=tf.nn.relu,
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                name="first_dense_layer",
                )

        out = \
            tf.layers.dense(
                out,
                4,
                activation=None, #Linear, because they're logits!
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                name="second_dense_layer",
                )

        logits_out = out
        self.logits_out = logits_out


    def _intense_create_prediction_model(self):
        """
        We want this to have a wide timestep-range. So, I think that something like 20 is good.

        We also want the inputs to make sense. So, let's use batch_norm. Because I'm not sure about
        the sizes of these things.

        Finally, we want to scale it down. With max_pool_1d. That should be all we really need.

        It's gonna be a pretty big model, unfortunately. Let's say the initial thing takes in 20
        timesteps, and maps it to 50 spots. That's 24*20*40 = 19,200 which is about 20,000.
        We only have like 2100 samples to train off of. I smell overfitting. We'll just regularize
        heavily... Honestly, it shouldn't be a hard problem anyways.

        KERNELS ARE USUALLY BIGGER THAN STRIDE. So, we'll say, kernel takes up 5, stride takes up 3.
        That way, it divides by 3 every time. Obviously.

        So, it should map to 50 channels, with "SAME" padding. Then, it should max_pool down a factor
        of 3. Then, it should map to 50 channels. Then, it should max_pool down a factor of 3.

        So, conv1d that maps to 50 channels. Then, max_pool_1d to
        Let's start with conv1d, then max_pool_1d, then conv_1d, then max_pool_1d

        It's gonna take absolutely forever if it tries to do a conv on every single timestep.
        Instead, I'll skip around, every 5. That gives lots of overlap.

        """

        out = self.input_ph
        # https://www.tensorflow.org/api_docs/python/tf/layers/conv1d
        out = \
            tf.layers.conv1d(
                out,
                50,
                kernel_size=20,
                strides=5,
                padding="VALID",
                activation=tf.nn.relu,
                data_format="channels_first", # IMPORTANT!!!
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                name="first_conv_layer",
                )

        out_shape = out.get_shape()
        print("I think that the first out shape should be (-1, 50, 95) or something")
        print("It is {}".format(out_shape))
        assert out_shape[1] == 50

        out = tf.layers.max_pooling1d(out,
                                   pool_size=5,
                                   strides=3,
                                   padding="VALID",
                                   data_format="channels_first"
                                   )

        out_shape = out.get_shape()
        print("Now, second out should have smaller last dimension, by a lot. Like (-1, 50, 30) or something")
        print("It is {}".format(out_shape))
        assert out_shape[1] == 50

        out = \
            tf.layers.conv1d(
                out,
                50,
                kernel_size=10,
                strides=3,
                padding="VALID",
                activation=tf.nn.relu,
                data_format="channels_first", # IMPORTANT!!!
                kernel_regularizer= tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                name="second_conv_layer",
                )

        out_shape = out.get_shape()
        print("I think that the first out shape should be (-1, 50, 10) or something")
        print("It is {}".format(out_shape))
        assert out_shape[1] == 50
        assert out_shape[2] == 8

        out = tf.reshape(out, 400) #Flattening the tensor.

        out = \
            tf.layers.dense(
                out,
                50,
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                bias_regularizer=tf.contrib.layers.l2_regularizer(scale=self.scale_conv_regularizer),
                name="first_dense_layer",
                )


if __name__ == '__main__':
    SpiderModel()
