import tensorflow as tf
import numpy as np
from alexnet import AlexNet 

from load_data import get_data
from load_data import get_batch_data

train_data, test_data, train_label, test_label = get_data();


learning_rate = 0.01
num_epochs = 10
batch_size = 128



# Network params
dropout = 0.5
num_classes = 9
train_layers = ['fc8', 'fc7', 'fc6']

x = tf.placeholder(tf.float32, [batch_size, 227, 227, 3])
y = tf.placeholder(tf.float32, [batch_size, num_classes])
keep_prob = tf.placeholder(tf.float32)

# Initialize model
model = AlexNet(x, keep_prob, num_classes, train_layers)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = model.fc8, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(model.fc8,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


with tf.Session() as sess:

    
    sess.run(tf.global_variables_initializer())

    step = 1;
    print(step,batch_size,num_epochs);
    while step < num_epochs:
        batch_xs , batch_ys = get_batch_data (batch_size)
        
        # batch_xs, batch_ys = next_batch(batch_size)
        # 获取批数据
       
        sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, keep_prob: dropout})
        if step % display_step == 0:
            # 计算精度
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            # 计算损失值
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print ("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy = " + "{:.5f}".format(acc))
        step += 1
    print ("Optimization Finished!")










