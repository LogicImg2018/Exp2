# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
from alexnet import AlexNet 
from datetime import datetime
from load_data import Load_data

from result import Result

data = Load_data("training1","testing")

result = Result("testing")

x_test = data.get_test_data()

x_train, y_train = data.get_data();


a = x_train[0];
b = y_train[0];
mean_px = x_train.mean().astype(np.float32)
std_px = x_train.std().astype(np.float32)

x_train =( x_train - mean_px)/std_px

permutation = np.random.permutation(y_train.shape[0])
train_data  = x_train[permutation, :, :]
train_label = y_train[permutation]

acm = np.size(train_label,0)


a=[]

# Network params
learning_rate = 0.01
num_epochs = 100
batch_size = 125
display_step = 4;
dropout = 0.5
num_classes = 50
train_layers = ['fc8']

print(batch_size)

x = tf.placeholder(tf.float32, [ batch_size,227, 227, 3])
y = tf.placeholder(tf.float32, [ batch_size,num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model.fc8, labels = y))

a = tf.global_variables()[-2:]

print(a)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost,var_list = a)

correct_pred = tf.equal(tf.argmax(model.fc8,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
saver = tf.train.Saver()



with tf.Session() as sess:

    # saver = tf.train.import_meta_graph('Model/model.ckpt.meta')
    # saver.restore(sess, tf.train.latest_checkpoint("Model/"))
    sess.run(tf.global_variables_initializer())
    model.load_initial_weights(sess)

    print("{} Start training...".format(datetime.now()))
    # print("{} Open Tensorboard at --logdir {}".format(datetime.now(),
    #                                                   filewriter_path))


    step = 0;
    loop = 0;
    print(step,batch_size,num_epochs);
    while step < num_epochs:
        begin = (batch_size*loop)
        end = (batch_size*(loop+1))

        if end>=acm:
            begin = 0;
            end = batch_size;
            loop = 0;

        batch_xs =  train_data[begin:end] 
        batch_ys = train_label[begin:end] 
        # print(np.shape(batch_xs))
        # print(np.shape(batch_ys))

        # batch_xs, batch_ys = next_batch(batch_size)
        # 获取批数据
       
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        # sess.run(train_data,train_label)
        
        if step % 4 == 0:
            permutation = np.random.permutation(y_train.shape[0])
            train_data  = x_train[permutation, :, :]
            train_label = y_train[permutation]

        if step % display_step == 0:


            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step += 1

        if step % 5 == 0:

            step_pre = 0;
            max_step = 2500.0/batch_size
            pre = model.fc8.eval(feed_dict={x:x_test[0:batch_size],keep_prob:1.})

            for i in range(2,int(max_step)):
                # print(i)
                pre_1 = model.fc8.eval(feed_dict={x:x_test[batch_size*(i-1):batch_size*i],keep_prob:1.})
                pre = np.concatenate((pre,pre_1),axis = 0)

            print(step)
            result.get_result(pre);
        if step % 5 == 0:
            saver.save(sess,"Model/model.ckpt")




