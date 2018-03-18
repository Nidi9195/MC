import sys
#import pickle as pk
import numpy as np
import tensorflow as tf
import sklearn.metrics as sm
import self_read_data as rd

print("Done importing")

batch_size    = 4
learning_rate = 0.003
n_epoch       = 50
n_total       = 1000
n_samples     = 600                              # change to 1000 for entire dataset
cv_split      = 0.8                             
train_size    = int(n_samples * cv_split)                               
test_size     = n_samples - train_size

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def init_biases(shape):
    return tf.Variable(tf.zeros(shape))

def cnn(melspectrogram, weights, phase_train):

    x = tf.reshape(melspectrogram,[-1,1,96,1366])
    x = batch_norm(melspectrogram, 1366, phase_train)
    x = tf.reshape(melspectrogram,[-1,96,1366,1])
    conv2_1 = tf.add(tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv1'])
    conv2_1 = tf.nn.relu(batch_norm(conv2_1, 32, phase_train))
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2 = tf.add(tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv2'])
    conv2_2 = tf.nn.relu(batch_norm(conv2_2, 128, phase_train))
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3 = tf.add(tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv3'])
    conv2_3 = tf.nn.relu(batch_norm(conv2_3, 128, phase_train))
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4 = tf.add(tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv4'])
    conv2_4 = tf.nn.relu(batch_norm(conv2_4, 192, phase_train))
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5)

    conv2_5 = tf.add(tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME'), weights['bconv5'])
    conv2_5 = tf.nn.relu(batch_norm(conv2_5, 256, phase_train))
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    p_y_X = tf.nn.sigmoid(tf.add(tf.matmul(flat,weights['woutput']),weights['boutput']))

    return p_y_X


def sort_result(tags, preds):
    result = zip(tags, preds)
    sorted_result = sorted(result, key=lambda x: x[1], reverse=True)
    return [(name, '%5.3f' % score) for name, score in sorted_result]

if __name__ == '__main__':
    print("Beginning")
    #indices = np.arange(n_samples)
    #indices = np.arange(n_total)
    #np.random.shuffle(indices)
    indices = np.random.randint(n_total, size=n_samples)
    train_indices = indices[0:train_size]
    test_indices  = indices[train_size:]
    print("Indices ready")

    labels = rd.get_labels()

    X_test = rd.get_melspectrograms_indexed(test_indices)
    y_train = labels[train_indices]
    y_test = labels[test_indices]
    print("y ready")
    
    weights = {
        'wconv1':init_weights([3, 3, 1, 32]),
        'wconv2':init_weights([3, 3, 32, 128]),
        'wconv3':init_weights([3, 3, 128, 128]),
        'wconv4':init_weights([3, 3, 128, 192]),
        'wconv5':init_weights([3, 3, 192, 256]),
        'bconv1':init_biases([32]),
        'bconv2':init_biases([128]),
        'bconv3':init_biases([128]),
        'bconv4':init_biases([192]),
        'bconv5':init_biases([256]),
        'woutput':init_weights([256, 10]),
        'boutput':init_biases([10])}

    X = tf.placeholder("float", [None, 96, 1366, 1])
    y = tf.placeholder("float", [None, 10])
    lrate = tf.placeholder("float")

    x = tf.reshape(X,[-1,96,1366,1])
    conv2_1_1 = tf.nn.conv2d(x, weights['wconv1'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_1 = tf.nn.relu(conv2_1_1 + weights['bconv1'])
    mpool_1 = tf.nn.max_pool(conv2_1, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_1 = tf.nn.dropout(mpool_1, 0.5)

    conv2_2_1 = tf.nn.conv2d(dropout_1, weights['wconv2'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_2 = tf.nn.relu(conv2_2_1 + weights['bconv2'])
    mpool_2 = tf.nn.max_pool(conv2_2, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_2 = tf.nn.dropout(mpool_2, 0.5)

    conv2_3_1 = tf.nn.conv2d(dropout_2, weights['wconv3'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_3 = tf.nn.relu(conv2_3_1 + weights['bconv3'])
    mpool_3 = tf.nn.max_pool(conv2_3, ksize=[1, 2, 4, 1], strides=[1, 2, 4, 1], padding='VALID')
    dropout_3 = tf.nn.dropout(mpool_3, 0.5)

    conv2_4_1 = tf.nn.conv2d(dropout_3, weights['wconv4'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_4 = tf.nn.relu(conv2_4_1 + weights['bconv4'])
    mpool_4 = tf.nn.max_pool(conv2_4, ksize=[1, 3, 5, 1], strides=[1, 3, 5, 1], padding='VALID')
    dropout_4 = tf.nn.dropout(mpool_4, 0.5) #Shape:[4, 1, 14, 128]

    conv2_5_1 = tf.nn.conv2d(dropout_4, weights['wconv5'], strides=[1, 1, 1, 1], padding='SAME')
    conv2_5 = tf.nn.relu(conv2_5_1 + weights['bconv5'])
    mpool_5 = tf.nn.max_pool(conv2_5, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='VALID')
    dropout_5 = tf.nn.dropout(mpool_5, 0.5)

    flat = tf.reshape(dropout_5, [-1, weights['woutput'].get_shape().as_list()[0]])
    y_ = tf.nn.sigmoid(tf.add(tf.matmul(flat,weights['woutput']),weights['boutput']))
        
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_, labels=y))
    train_op = tf.train.RMSPropOptimizer(learning_rate, 0.9).minimize(cost)
    #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
    #train_op = optimizer.minimize(cost)
    predict_op = y_

    tags = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

    with tf.Session() as sess:
        #tf.initialize_all_variables().run()
        tf.global_variables_initializer().run()
        for i in range(n_epoch):
            print("Epoch",i)
            training_batch = zip(range(0, train_size, batch_size),range(batch_size, train_size+1, batch_size))
            for start, end in training_batch:
                X_train = rd.get_melspectrograms_indexed(train_indices[start:end])
                train_input_dict = {X: X_train, 
                                    y: y_train[start:end],
                                    lrate: learning_rate
                                    }
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
