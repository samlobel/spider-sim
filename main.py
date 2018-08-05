from learning_model import SpiderModel
from spider_data import get_train_data, get_test_data

import tensorflow as tf



SIZE_TRAIN = 2100 # I think
BATCHES_PER_EPOCH = 10
assert SIZE_TRAIN % BATCHES_PER_EPOCH == 0

batch_size = int(SIZE_TRAIN / BATCHES_PER_EPOCH)

train_dataset = get_train_data(batch_size)
test_dataset = get_test_data(700)

iterator = tf.data.Iterator.from_structure(train_dataset.output_types,
                                               train_dataset.output_shapes)

next_element = iterator.get_next()

# samples_ph = tf.placeholder(dtype=tf.float32, shape=[None, 24, 500])
# targets_ph = tf.placeholder(dtype=tf.int64, shape=[None])
# model = SpiderModel(samples=samples_ph, targets=targets_ph) #THIS WORKS!

model = SpiderModel(samples=next_element[0],
                    targets=next_element[1],
                    lr=1e-4,
)


training_init_op = iterator.make_initializer(train_dataset)
testing_init_op = iterator.make_initializer(test_dataset)

init_op = tf.global_variables_initializer()

# log_dir = './logging/{}'.format(DATA_SUBDIR, arch_str)
log_dir = "./logging/logging4"
# if os.path.exists(log_dir):
#     shutil.rmtree(log_dir)
print("log directory: %s" % log_dir)
summary_writer = tf.summary.FileWriter(log_dir, graph=tf.get_default_graph())


with tf.Session() as sess:
    sess.run(init_op)
    # model.saver.restore(sess, 'checkpoints/model')

    for epoch in range(1000):
        sess.run(training_init_op)
        for i in range(BATCHES_PER_EPOCH):
            global_step = epoch*BATCHES_PER_EPOCH + i
            # print("EPOCH {} BATCH {}".format(epoch, i))
            _, summary = sess.run([model.train_op, model.training_summaries])
            summary_writer.add_summary(summary, global_step=global_step)
            summary_writer.flush() #At the end of an epoch I would like to see whats going on...

        print("Running test epoch {}".format(epoch))
        sess.run(testing_init_op)
        summary = sess.run(model.testing_summaries, feed_dict={model.is_training: False})
        summary_writer.add_summary(summary, global_step=epoch)

    model.saver.save(sess, 'checkpoints/model')
