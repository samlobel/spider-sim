# exit("Just so I don't run int accidentally...")

from locating_learning_model import SpiderLocator
from spider_data import get_locating_train_data, get_locating_test_data

import tensorflow as tf
import numpy as np



# SIZE_TRAIN = 2100 # I think
# BATCHES_PER_EPOCH = 10
# assert SIZE_TRAIN % BATCHES_PER_EPOCH == 0

# batch_size = int(SIZE_TRAIN / BATCHES_PER_EPOCH)

# batch_size = 100

MEAN_DATA = np.asarray([
    4.4874486e-02, 4.5664940e-02, 8.3224371e-02, 1.6056228e+00, 1.6082847e+00,
    1.1395665e+03, 4.4821296e-02, 4.5671549e-02, 1.4073811e-02, 1.3133851e+03,
    1.3382377e+03, 2.0117255e+02, 4.7401603e-02, 4.8398234e-02, 1.6391309e-02,
    1.8579895e+00, 1.9230013e+00, 2.0557390e+02, 4.7273219e-02, 4.8525523e-02, 
    1.6552730e-02, 1.8577598e+00, 1.9231324e+00, 2.0168808e+02, 4.9615213e-01
])

VARIANCE_DATA = np.asarray([
    5.0566737e-03, 5.1953350e-03, 3.0809554e-01, 8.8558588e+00, 8.9578514e+00,
    4.6601388e+07, 5.0339620e-03, 5.1828744e-03, 3.2287095e-02, 1.4097924e+07,
    1.4427968e+07, 4.4046120e+06, 4.9797399e-03, 5.0399355e-03, 3.4936607e-02,
    9.3696833e+00, 9.7159138e+00, 4.4643995e+06, 4.9802926e-03, 5.0624828e-03,
    3.5007007e-02, 9.4073076e+00, 9.6957226e+00, 4.3708295e+06, 6.1335776e-02
])

train_dataset = get_locating_train_data(6250)
test_dataset = get_locating_test_data(6250)



iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

next_element = iterator.get_next()

# samples_ph = tf.placeholder(dtype=tf.float32, shape=[None, 24, 500])
# targets_ph = tf.placeholder(dtype=tf.int64, shape=[None])
# model = SpiderModel(samples=samples_ph, targets=targets_ph) #THIS WORKS!

model = SpiderLocator(samples=next_element[0],
                    targets=next_element[1],
                    mean_samples=MEAN_DATA,
                    variance_samples=VARIANCE_DATA,
                    # lr=1e-4,
                    # scale_conv_regularizer=1e-3
)


training_init_op = iterator.make_initializer(train_dataset)
testing_init_op = iterator.make_initializer(test_dataset)

init_op = tf.global_variables_initializer()

# log_dir = './logging/{}'.format(DATA_SUBDIR, arch_str)
log_dir = "./logging/logging6/big_network_adam_2"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)
print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init_op)
    # model.saver.restore(sess, 'checkpoints/model')

    global_step=0
    try:
        for epoch in range(10000):
            sess.run(training_init_op)
            batch_num = 0
            while True:
                try:
                    global_step += 1
                    batch_num += 1
                    # for i in range(BATCHES_PER_EPOCH):
                    # global_step = epoch*BATCHES_PER_EPOCH + i
                    # print("EPOCH {} BATCH {}".format(epoch, batch_num))
                    _, summary = sess.run([model.train_op, model.training_summaries])
                    summary_writer.add_summary(summary, global_step=global_step)
                    summary_writer.flush() #At the end of an epoch I would like to see whats going on...
                except tf.errors.OutOfRangeError:
                    # print('breaking.')
                    break
            print("Running test epoch {}".format(epoch))
            sess.run(testing_init_op)
            summary = sess.run(model.testing_summaries)#, feed_dict={model.is_training: False})
            summary_writer.add_summary(summary, global_step=epoch)
    except KeyboardInterrupt:
        print("Exiting, but saving first...")
        pass
    model.saver.save(sess, 'checkpoints/locator/locate_prey_model')
