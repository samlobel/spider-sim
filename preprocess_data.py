"""
I expect that this is going to be ugly the first time through. Anyways, what does it need to know?

It needs to know the input dimensions.

It needs to know the targets for each sample.
It needs to have a good way of reading the data, and shuffling it, and whatnot.

It should have a way to divide it into train and test BEFORE we start.

Should probably use np.concatenate
"""

import os
import pickle

import numpy as np

def munge_data():
    """
    Returns, un-mixed, all the datas we made.
    """

    sample_names = ['data/train_samples_radial_before.npy',
                    'data/train_samples_radial_after.npy',
                    'data/train_samples_azimuthal_before.npy',
                    'data/train_samples_azimuthal_after.npy']

    target_names = ['data/train_targets_radial_before.npy',
                    'data/train_targets_radial_after.npy',
                    'data/train_targets_azimuthal_before.npy',
                    'data/train_targets_azimuthal_after.npy']

    samples = [np.load(name) for name in sample_names]
    assert len(set([s.shape for s in samples])) == 1 # All the same size.
    s_shape = samples[0].shape
    samples = np.concatenate(samples)
    assert s_shape[1:] == samples.shape[1:] #Shape is same after first dimension.

    targets = [np.load(name) for name in target_names]
    assert len(set([t.shape for t in targets])) == 1 # All the same size.
    t_shape = targets[0].shape
    targets = np.concatenate(targets)
    assert t_shape[1:] == targets.shape[1:] #Shape is same after first dimension.

    assert samples.shape[0] == targets.shape[0]
    print("Number of samples, targets: {}".format(samples.shape[0]))
    print('shape of samples: {}'.format(samples.shape))
    print('shape of targets: {}'.format(targets.shape))

    return samples, targets

def mix_up_samples(samples, targets):
    """
    Randomly permutes both samples and targets.

    https://stackoverflow.com/questions/4601373/better-way-to-shuffle-two-numpy-arrays-in-unison
    """

    assert len(samples) == len(targets)
    p = np.random.permutation(len(samples))
    return samples[p], targets[p]

def split_into_train_and_test(samples, targets, proportion_train=0.9):
    # Re-mix them just in case.
    samples, targets = mix_up_samples(samples, targets)
    assert len(samples) == len(targets)
    assert 0 < proportion_train < 1
    num_samples_train = int(len(samples)*proportion_train)
    samples_train, targets_train = samples[:num_samples_train], targets[:num_samples_train]
    samples_test, targets_test = samples[num_samples_train:], targets[num_samples_train:]

    assert len(samples_train) == len(targets_train)
    assert len(samples_test) == len(targets_test)

    return {
        'train': (samples_train, targets_train),
        'test': (samples_test, targets_test),
        'proportion_train': proportion_train,
    }

def write_data_to_final(proportion_train=0.9, overwrite=False):
    print("Going to write the train-test split now, to a file called data.pkl")
    if not overwrite and os.path.exists("data/data.pkl"):
        print("Going to exit early, because it already exists.")
        return


    samples, targets = munge_data()
    samples_targets = mix_up_samples(samples, targets)
    data = split_into_train_and_test(samples, targets, proportion_train=proportion_train)
    with open('data/data.pkl', 'wb') as f:
        print('dumping')
        pickle.dump(data, f)
        print('dumped')



if __name__ == '__main__':
    write_data_to_final(overwrite=True, proportion_train=0.75)
