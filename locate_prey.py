# exit("Just so I don't run int accidentally...")

from locating_learning_model import SpiderLocator
from spider_data import get_locating_train_data, get_locating_test_data

import tensorflow as tf



# SIZE_TRAIN = 2100 # I think
# BATCHES_PER_EPOCH = 10
# assert SIZE_TRAIN % BATCHES_PER_EPOCH == 0

# batch_size = int(SIZE_TRAIN / BATCHES_PER_EPOCH)

# batch_size = 100

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
                    # lr=1e-4,
                    # scale_conv_regularizer=1e-3
)


training_init_op = iterator.make_initializer(train_dataset)
testing_init_op = iterator.make_initializer(test_dataset)

init_op = tf.global_variables_initializer()

# log_dir = './logging/{}'.format(DATA_SUBDIR, arch_str)
log_dir = "./logging/logging6"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)
print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init_op)
    # model.saver.restore(sess, 'checkpoints/model')

    global_step=0
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

    model.saver.save(sess, 'checkpoints/locator/model2')
