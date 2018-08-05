"""
Here, we'll make a train/test loader for the python data.

http://adventuresinmachinelearning.com/tensorflow-dataset-tutorial/

This is the only place I found a way to do samples/labels at once.

https://www.tensorflow.org/guide/datasets
This gives good instruction on how to replace training data with testing data in your pipeline...

NOTE: It's going to be tough to use this dataset nonsense, along with actually evaluating against
real stuff. Because when we're doing real stuff, we don't have a target usually.

BUT, I think we'll be able to just pass real data to overwrite some intermediary node, so it
may actually be alright. I'm going to try this out, in an effort to learn something new.

So, I have this iterator. I'm going to make two, and use the from_structure thing.



"""

import pickle
import tensorflow as tf


with open('data/data.pkl', 'rb') as f:
    datasets = pickle.load(f)

# import ipdb; ipdb.set_trace()
train_data, test_data = datasets['train'], datasets['test']


def get_data(input_dataset, batch_size):
    samples, targets = input_dataset
    samples_t = tf.data.Dataset.from_tensor_slices(samples)
    targets_t  = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((samples_t, targets_t)).shuffle(500).repeat().batch(batch_size)#.batch_and_drop_remainder(batch_size)

    return dataset

def get_train_data(batch_size):
    return get_data(train_data, batch_size)

def get_test_data(batch_size):
    return get_data(test_data, batch_size)


if __name__ == '__main__':
    asdf = get_train_data(10)
    import ipdb; ipdb.set_trace()
