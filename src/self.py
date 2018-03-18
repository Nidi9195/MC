import sys
import pickle as pk
import numpy as np
import tensorflow as tf
import self_read_data as rd

''' Check 1D conv and 1D max pooling
Check gru1 as gru vs dynamic rnn '''

batch_size    = 4
learning_rate = 0.005
n_epoch       = 50
n_samples     = 100                             
cv_split      = 0.7                             
train_size    = int(n_samples * cv_split)                               
test_size     = n_samples - train_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

#xx = tf.placeholder(tf.float, [None, 96, 1366, 1])
#x = tf.reshape(xx,[-1,96,1366,1])

if __name__ == '__main__':

    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    train_indices = indices[0:train_size]
    test_indices  = indices[train_size:]

    labels = rd.get_labels()
    #1000 x 10 one hot encoded in order

    X_test = rd.get_melspectrograms_indexed(test_indices) #returns [test_indices_size, 96, 1366, 1]
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    print("Y done")

    weights = {
        'wconv1':init_weights([3, 3, 1, 64]),
        'wconv2':init_weights([3, 3, 64, 128]),
        'wconv3':init_weights([3, 3, 128, 128]),
        'wconv4':init_weights([3, 3, 128, 128]),
        'bconv1':init_biases([64]),
        'bconv2':init_biases([128]),
        'bconv3':init_biases([128]),
        'bconv4':init_biases([128]),
        'woutput':init_weights([32, 10]),
        'boutput':init_biases([10])}

    X = tf.placeholder("float", [None, 96, 1366, 1])
    y = tf.placeholder("float", [None, 10])
    lrate = tf.placeholder("float")
    phase_train = tf.placeholder(tf.bool, name='phase_train')

    x = tf.reshape(X,[-1,96,1366,1])

    conv2_1_1 = tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_1 = tf.nn.relu(conv2_1_1 + weights['bconv1'])
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2_1 = tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_2 = tf.nn.relu(conv2_2_1 + weights['bconv2'])
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 3, 3, 1], strides=[1, 3, 3, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3_1 = tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.relu(conv2_3_1 + weights['bconv3'])
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 3, 3, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4_1 = tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_4 = tf.nn.relu(conv2_4_1 + weights['bconv4'])
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5) #Shape:[4, 1, 14, 128]

    gru1_in = tf.reshape(dropout_4,[-1, 14, 128])
    #gru1 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 15)
    #gru1 = tf.contrib.rnn.GRUCell(32)
    stacked_rnn = []
    for i in range(14):
        stacked_rnn.append(tf.nn.rnn_cell.GRUCell(32)) 
    gru1 = tf.nn.rnn_cell.MultiRNNCell( stacked_rnn )
    gru1_out, state = tf.nn.dynamic_rnn (gru1, gru1_in, dtype=tf.float32, scope='gru1')

    gru2 = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.GRUCell(32)] * 14)
    gru2_out, state = tf.nn.dynamic_rnn(gru2, gru1_out, dtype=tf.float32, scope='gru2')
    gru2_out = tf.transpose(gru2_out, [1, 0, 2])
    gru2_out = tf.gather(gru2_out, int(gru2_out.get_shape()[0]) - 1)
    dropout_5 = tf.nn.dropout(gru2_out, 0.3)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(flat,weights['woutput']),weights['boutput']))
    
    #y_ = x
    #y_ = crnn(X, weights, phase_train)

    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    predict_op = y_

    tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']
    print(tags)

    with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        for i in range(n_epoch):
            print("Epoch ",i)
            training_batch = zip(range(0, train_size, batch_size),range(batch_size, train_size+1, batch_size))            
            for start, end in training_batch:
                X_train = rd.get_melspectrograms_indexed(train_indices[start:end])
                train_input_dict = {X: X_train, 
                                    y: y_train[start:end],
                                    lrate: learning_rate,
                                    phase_train: True}
                #m = sess.run(tf.shape(gru2_out), feed_dict={X:X_train})
                #print("SHAPE!",m)
                #sys.exit()
                sess.run(train_op, feed_dict=train_input_dict)
            mm = sess.run([cost], feed_dict=train_input_dict)
            print("Cost:",mm)

            test_indices = np.arange(len(X_test))
            np.random.shuffle(test_indices)
            test_indices = test_indices[0:test_size]
            #X_test = rd.get_melspectrograms_indexed(test_indices)

            test_input_dict = {X: X_test[test_indices],
                               y: y_test[test_indices]
                               }
            predictions = sess.run(predict_op, feed_dict=test_input_dict)
            print('Epoch : ', i,  'AUC : ', sm.roc_auc_score(y_test[test_indices], predictions, average='samples'))
            #print(i, np.mean(np.argmax(y_test[test_indices], axis=1) == predictions))
            #print(sort_result(tags, predictions)[:5])

            
        

    
