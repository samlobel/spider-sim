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
from sklearn.model_selection import train_test_split


with open('data/data.pkl', 'rb') as f:
    datasets = pickle.load(f)

# import ipdb; ipdb.set_trace()
# These are numpy arrays I believe.
train_data, test_data = datasets['train'], datasets['test']


def get_data(input_dataset, batch_size):
    # This works for the locating data too! Very nice.
    samples, targets = input_dataset
    samples_t = tf.data.Dataset.from_tensor_slices(samples)
    targets_t  = tf.data.Dataset.from_tensor_slices(targets)
    dataset = tf.data.Dataset.zip((samples_t, targets_t)).shuffle(500).repeat().batch(batch_size, drop_remainder=True)

    return dataset

def get_train_data(batch_size):
    return get_data(train_data, batch_size)

def get_test_data(batch_size):
    return get_data(test_data, batch_size)


"""
Okay, so I'm going to pickle the data. I'll have it be a dictionary to inputs, targets.
In numpy, I'll get a permutation, and then split it up by train and test.
I'll do the splitting with scikit-learn, cause it's easiest.

"""

# def get_data_locating(inputs, targets, batch_size):
#     samples_t = tf.data.Dataset.from_tensor_slices(samples)
#     samples_t = tf.data.Dataset.from_tensor_slices(samples)


def get_train_test_split_for_locating_data(file_location, test_proportion=0.1, whiten_input_data=True):
    with open(file_location, 'rb') as f:
        data_dict = pickle.load(f)

    X_train, X_test, y_train, y_test = train_test_split(data_dict['inputs'],
                                                        data_dict['targets'],
                                                        test_size=test_proportion,
                                                        random_state=42)
    
    return {
        'train': (X_train, y_train),
        'test': (X_test, y_test),
    }

locating_data = get_train_test_split_for_locating_data("data/locating/data.pkl")

def get_locating_train_data(batch_size):
    return get_data(locating_data['train'], batch_size)

def get_locating_test_data(batch_size):
    return get_data(locating_data['test'], batch_size)


if __name__ == '__main__':
    # asdf = get_locating_train_data(10)
    # import ipdb ; ipdb.set_trace()
    asdf = get_train_data(10)
    import ipdb; ipdb.set_trace()
