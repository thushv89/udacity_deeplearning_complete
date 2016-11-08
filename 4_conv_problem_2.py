# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np


image_size = 28
num_labels = 10
num_channels = 1 # grayscale
batch_size = 16
patch_size = 5
depth = 16
num_steps = 20001

start_lr = 0.05
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = True

early_stopping = True

#seems having SQRT(2/n_in) as the stddev gives better weight initialization
#I used constant of 0.1 and tried weight_init_factor=0.01 but non of them worked
#lowering this makes the CNN impossible to learn
weight_init_factor = 1

beta = 1e-10
#--------------------- SUBSAMPLING OPERATIONS and THERE PARAMETERS -------------------------------------------------#
conv_ops = ['conv_1','pool_1','conv_2','pool_1','conv_3','pool_1','fulcon_hidden_1','fulcon_hidden_2','fulcon_out']

#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
conv_1_hyparams = {'weights':[3,3,num_channels,depth],'stride':[1,1,1,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[3,3,depth,depth],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[3,3,depth,depth],'stride':[1,1,1,1],'padding':'SAME'}
pool_1_hyparams = {'type':'max','kernel':[1,2,2,1],'stride':[1,2,2,1],'padding':'SAME'}
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':512,'out':10}

hyparams = {'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3': conv_3_hyparams,
            'pool_1': pool_1_hyparams, 'fulcon_hidden_1':hidden_1_hyparams,
            'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams}
#=====================================================================================================================#
train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

tf_dataset = None
tf_labels = None

weights,biases = {},{}

def load_data():
    global train_dataset,train_labels
    global valid_dataset,valid_labels
    global test_dataset,test_labels
    pickle_file = 'notMNIST.pickle'

    with open(pickle_file, 'rb') as f:
        save = pickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memory
        print('Training set', train_dataset.shape, train_labels.shape)
        print('Validation set', valid_dataset.shape, valid_labels.shape)
        print('Test set', test_dataset.shape, test_labels.shape)


def reshape_data():
    global train_dataset,train_labels
    global valid_dataset,valid_labels
    global test_dataset,test_labels

    def reformat(dataset, labels):
        dataset = dataset.reshape((-1, image_size, image_size, num_channels)).astype(np.float32)
        labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
        return dataset, labels

    train_dataset, train_labels = reformat(train_dataset, train_labels)
    valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
    test_dataset, test_labels = reformat(test_dataset, test_labels)
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


def create_subsample_layers():
    print('Defining parameters ...')

    for op in conv_ops:
        if 'fulcon' in op:
            #we don't create weights biases for fully connected layers because we need to calc the
            #fan_out of the last convolution/pooling (subsampling) layer
            #as that's gonna be fan_in for the 1st hidden layer
            break
        if 'conv' in op:
            print('\tDefining weights and biases for %s (weights:%s)'%(op,hyparams[op]['weights']))
            print('\t\tWeights:%s'%hyparams[op]['weights'])
            print('\t\tBias:%d'%hyparams[op]['weights'][3])
            weights[op]=tf.Variable(
                tf.truncated_normal(hyparams[op]['weights'],
                                    stddev=2./(hyparams[op]['weights'][0]*hyparams[op]['weights'][0])
                                    )
            )
            biases[op] = tf.Variable(tf.constant(1.,shape=[hyparams[op]['weights'][3]]))

def create_fulcon_layers(fan_in):
    hyparams['fulcon_hidden_1']['in'] = fan_in
    for op in conv_ops:
            if 'fulcon' not in op:
                continue
            else:
                if op in weights and op in biases:
                    break

                weights[op] = tf.Variable(
                    tf.truncated_normal(
                        [hyparams[op]['in'],hyparams[op]['out']],stddev=2./hyparams[op]['in']
                    )
                )

                biases[op] = tf.Variable(tf.zeros(shape=[hyparams[op]['out']]))


def get_logits(dataset):

    # Variables.
    x = dataset
    print('Calculating inputs for data X(%s)...'%x.get_shape().as_list())
    for op in conv_ops:
        if 'conv' in op:
            print('\tCovolving data (%s)'%op)
            x = tf.nn.conv2d(x, weights[op], hyparams[op]['stride'], padding=hyparams[op]['padding'])

            x = tf.nn.relu(x + biases[op])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if 'pool' in op:
            print('\tPooling data')
            x = tf.nn.max_pool(x,ksize=hyparams[op]['kernel'],strides=hyparams[op]['stride'],padding=hyparams[op]['padding'])
            print('\t\tX after %s:%s'%(op,x.get_shape().as_list()))
        if 'fulcon' in op:
            break

    # we need to reshape the output of last subsampling layer to
    # convert 4D output to a 2D input to the hidden layer
    # e.g subsample layer output [batch_size,width,height,depth] -> [batch_size,width*height*depth]
    shape = x.get_shape().as_list()
    rows = shape[0]
    create_fulcon_layers(shape[1] * shape[2] * shape[3])
    print('Unwrapping last convolution layer %s to %s hidden layer'%(shape,(rows,hyparams['fulcon_hidden_1']['in'])))
    x = tf.reshape(x, [rows,hyparams['fulcon_hidden_1']['in']])

    for op in conv_ops:
        if 'fulcon_hidden' not in op:
            continue
        else:
            if use_dropout:
                x = tf.nn.dropout(tf.nn.relu(tf.matmul(x,weights[op])+biases[op]),keep_prob=1.-dropout_rate,seed=tf.set_random_seed(12321))
            else:
                x = tf.nn.relu(tf.matmul(x,weights[op])+biases[op])

    return tf.matmul(x, weights['fulcon_out']) + biases['fulcon_out']

def calc_loss(logits,labels):
    # Training computation.
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw else 0 for kw,w in weights.items()])

    return loss

def optimize_func(loss,global_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=500,decay_rate=0.99)
    else:
        learning_rate = start_lr

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer

def inc_global_step(global_step):
    return global_step.assign(global_step+1)

def predict_with_logits(logits):
    # Predictions for the training, validation, and test data.
    prediction = tf.nn.softmax(logits)
    return prediction

def predict_with_dataset(dataset):
    prediction = tf.nn.softmax(get_logits(dataset))
    return prediction

if __name__=='__main__':
    global tf_train_dataset,tf_train_labels,tf_valid_dataset,tf_test_dataset
    load_data()
    reshape_data()
    graph = tf.Graph()

    valid_accuracies = []

    with tf.Session(graph=graph) as session:
        #tf.global_variables_initializer().run()
        # Input data.
        global_step = tf.Variable(0, trainable=False)

        print('Input data defined...\n')
        tf_dataset = tf.placeholder(
        tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_valid_dataset = tf.constant(valid_dataset)
        tf_test_dataset = tf.constant(test_dataset)

        create_subsample_layers()

        logits = get_logits(tf_dataset)
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)

        valid_pred = predict_with_dataset(tf_valid_dataset)
        test_pred = predict_with_dataset(tf_test_dataset)

        tf.initialize_all_variables().run()
        print('Initialized...')
        print('Batch size:',batch_size)
        print('Depth:',depth)
        print('Num Steps: ',num_steps)
        print('Decay Learning Rate: ',decay_learning_rate,', ',start_lr)
        print('Dropout: ',use_dropout,', ',dropout_rate)
        print('Early Stopping: ',early_stopping)
        print('Beta: ',beta)
        print('==================================================\n')
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, l, _,predictions,_ = session.run([logits,loss,optimize,pred,inc_gstep], feed_dict=feed_dict)

            if step % 50 == 0:
                print('Global step: %d'%global_step.eval())
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                valid_predictions = session.run([valid_pred])
                v_accuracy = accuracy(valid_predictions[0], valid_labels)
                print('Validation accuracy: %.1f%%' %v_accuracy)
                if early_stopping and step>500 \
                        and len(valid_accuracies)>0 and v_accuracy < np.mean(valid_accuracies[-10:])*0.9:
                    break
                valid_accuracies.append(v_accuracy)

        test_predictions = session.run([test_pred])
        print('Test accuracy: %.1f%%' % accuracy(test_predictions[0], test_labels))