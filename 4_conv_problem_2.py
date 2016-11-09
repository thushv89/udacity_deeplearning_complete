# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import numpy as np
import os


'''=========================================================================
Batch size: 16
Depths:  {'conv_3': 32, 'conv_2': 64, 'iconv_1x1': 16, 'conv_1': 32, 'iconv_3x3': 16, 'iconv_5x5': 16}
Num Steps:  10001
Decay Learning Rate:  True ,  0.1
Dropout:  False ,  0.25
Early Stopping:  False
Include L2, Beta:  True ,  1e-10

So far worked (Normal CNN)
'conv_1'      'pool_1'     'conv_2'     'pool_1'    'conv_3'     'pool_2'     'fulcon_hidden_1','fulcon_out'
conv (3x3) > pool (2x2) > conv (3x3) > pool (2x2) > conv (3x3) > pool (2x2) (Subsampling layers)
1024->512->10 (hidden layers)

Inception CNN
'conv_1','pool_1','conv_2','pool_1','incept_1','pool_2','fulcon_hidden_1','fulcon_out'
=========================================================================='''

datatype = 'cifar-10'
if datatype=='cifar-10':
    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb
elif datatype=='notMNIST':
    image_size = 28
    num_labels = 10
    num_channels = 1 # grayscale

batch_size = 16
patch_size = 3

num_steps = 50001

start_lr = 0.1
decay_learning_rate = True

#dropout seems to be making it impossible to learn
#maybe only works for large nets
dropout_rate = 0.25
use_dropout = True

early_stopping = True
accuracy_drops_cap = 10

#seems having SQRT(2/n_in) as the stddev gives better weight initialization
#I used constant of 0.1 and tried weight_init_factor=0.01 but non of them worked
#lowering this makes the CNN impossible to learn
weight_init_factor = 1

include_l2_loss = True
beta = 1e-10

#making bias small seems to be helpful (pref 0)

#--------------------- SUBSAMPLING OPERATIONS and THERE PARAMETERS -------------------------------------------------#
conv_ops = ['conv_1','pool_1','conv_2','pool_2','conv_3','pool_2','incept_1','pool_3','fulcon_hidden_1','fulcon_hidden_2','fulcon_out']

#number of feature maps for each convolution layer
depth_conv = {'conv_1':128,'conv_2':64,'conv_3':32,'iconv_1x1':16,'iconv_3x3':16,'iconv_5x5':16}
incept_orders = {'incept_1':['ipool_2x2','iconv_1x1','iconv_3x3','iconv_5x5']}

#weights (conv): [width,height,in_depth,out_depth]
#kernel (pool): [_,width,height,_]
conv_1_hyparams = {'weights':[patch_size,patch_size,num_channels,depth_conv['conv_1']],'stride':[1,1,1,1],'padding':'SAME'}
conv_2_hyparams = {'weights':[patch_size,patch_size,depth_conv['conv_1'],depth_conv['conv_2']],'stride':[1,1,1,1],'padding':'SAME'}
conv_3_hyparams = {'weights':[patch_size,patch_size,depth_conv['conv_2'],depth_conv['conv_3']],'stride':[1,1,1,1],'padding':'SAME'}
pool_1_hyparams = {'type':'max','kernel':[1,2,2,1],'stride':[1,2,2,1],'padding':'SAME'}
pool_2_hyparams = {'type':'max','kernel':[1,2,2,1],'stride':[1,1,1,1],'padding':'SAME'}
pool_3_hyparams = {'type':'avg','kernel':[1,5,5,1],'stride':[1,2,2,1],'padding':'SAME'}
#I'm using only one inception module. Hyperparameters for the inception module found here
incept_1_hyparams = {
    'ipool_2x2':{'type':'avg','kernel':[1,5,5,1],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_1x1':{'weights':[1,1,depth_conv['conv_3'],depth_conv['iconv_1x1']],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_3x3':{'weights':[3,3,depth_conv['iconv_1x1'],depth_conv['iconv_3x3']],'stride':[1,1,1,1],'padding':'SAME'},
    'iconv_5x5':{'weights':[5,5,depth_conv['iconv_1x1'],depth_conv['iconv_5x5']],'stride':[1,1,1,1],'padding':'SAME'}
}

# fully connected layer hyperparameters
hidden_1_hyparams = {'in':0,'out':1024}
hidden_2_hyparams = {'in':1024,'out':512}
out_hyparams = {'in':512,'out':10}

hyparams = {'conv_1': conv_1_hyparams, 'conv_2': conv_2_hyparams, 'conv_3':conv_3_hyparams,
           'incept_1': incept_1_hyparams,'pool_1': pool_1_hyparams, 'pool_2':pool_2_hyparams, 'pool_3':pool_3_hyparams,
           'fulcon_hidden_1':hidden_1_hyparams,'fulcon_hidden_2': hidden_2_hyparams, 'fulcon_out':out_hyparams}
#=====================================================================================================================#

train_dataset, train_labels = None,None
valid_dataset, valid_labels = None,None
test_dataset, test_labels = None,None

tf_dataset = None
tf_labels = None

weights,biases = {},{}

valid_size,train_size,test_size = 0,0,0

def load_data_cifar10():

    cifar_file = 'data'+os.sep+'cifar_train.pickle'

    if os.path.exists(cifar_file):
        return

    train_pickle_file = 'data'+os.sep+'cifar_10_data_batch_'
    test_pickle_file = 'data' + os.sep + 'cifar_10_test_batch'
    train_raw = None
    test_dataset = None
    train_raw_labels = None
    test_labels = None

    #train data
    for i in range(1,5+1):
        with open(train_pickle_file+str(i),'rb') as f:
            save = pickle.load(f,encoding="latin1")

            if train_raw is None:
                train_raw = np.asarray(save['data'],dtype=np.float32)
                train_raw_labels = np.asarray(save['labels'],dtype=np.int16)
            else:

                train_raw = np.append(train_raw,save['data'],axis=0)
                train_raw_labels = np.append(train_raw_labels,save['labels'],axis=0)

    #test file
    with open(test_pickle_file,'rb') as f:
        save = pickle.load(f,encoding="latin1")
        test_dataset = np.asarray(save['data'],dtype=np.float32)
        test_labels = np.asarray(save['labels'],dtype=np.int16)

    valid_size_required = 10000
    valid_rand_idx = np.random.randint(0,train_raw.shape[0]-valid_size_required)
    valid_perm = np.random.permutation(train_raw.shape[0])[valid_rand_idx:valid_rand_idx+valid_size_required]

    valid_dataset = np.asarray(train_raw[valid_perm,:],dtype=np.float32)
    valid_labels = np.asarray(train_raw_labels[valid_perm],dtype=np.int16)
    print('Shape of valid dataset (%s) and labels (%s)'%(valid_dataset.shape,valid_labels.shape))

    train_dataset = np.delete(train_raw,valid_perm,axis=0)
    train_labels = np.delete(train_raw_labels,valid_perm,axis=0)
    print('Shape of train dataset (%s) and labels (%s)'%(train_dataset.shape,train_labels.shape))

    print('Per image whitening ...')
    pixel_depth = 255 if np.max(train_dataset[0,:])>1.1 else 1
    print('\tDectected pixel depth: %d'%pixel_depth)
    print('\tZero mean and Unit variance')
    train_dataset = np.subtract(train_dataset,np.mean(train_dataset,axis=1).reshape((-1,1)))/pixel_depth
    valid_dataset = np.subtract(valid_dataset,np.mean(valid_dataset,axis=1).reshape((-1,1)))/pixel_depth
    test_dataset = np.subtract(test_dataset,np.mean(test_dataset,axis=1).reshape((-1,1)))/pixel_depth
    print('\tTrain Mean/Variance:%.2f%.2f'%(
        np.mean(np.mean(train_dataset,axis=1),axis=0),
        np.mean(np.std(train_dataset,axis=1),axis=0))
          )
    print('\tValid Mean/Variance:%.2f%.2f'%(
        np.mean(np.mean(valid_dataset,axis=1),axis=0),
        np.mean(np.std(valid_dataset,axis=1),axis=0))
          )
    print('\tTest Mean/Variance:%.2f%.2f'%(
        np.mean(np.mean(test_dataset,axis=1),axis=0),
        np.mean(np.std(test_dataset,axis=1),axis=0))
          )
    print('Successfully whitened data ...\n')
    print('\nDumping processed data')
    cifar_data = {'train_dataset':train_dataset,'train_labels':train_labels,
                  'valid_dataset':valid_dataset,'valid_labels':valid_labels,
                  'test_dataset':test_dataset,'test_labels':test_labels
                  }
    try:
        with open(cifar_file, 'wb') as f:
            pickle.dump(cifar_data, f, pickle.HIGHEST_PROTOCOL)
    except Exception as e:
        print('Unable to save cifar_data:', e)


def reformat_data_cifar10():

    global train_dataset,train_labels,valid_dataset,valid_labels,test_dataset,test_labels
    global image_size,num_labels,num_channels
    global train_size,valid_size,test_size

    image_size = 32
    num_labels = 10
    num_channels = 3 # rgb

    print("Reformatting data ...")
    cifar10_file = 'data'+os.sep+'cifar_train.pickle'
    with open(cifar10_file,'rb') as f:
        save = pickle.load(f)
        train_dataset, train_labels = save['train_dataset'],save['train_labels']
        valid_dataset, valid_labels = save['valid_dataset'],save['valid_labels']
        test_dataset, test_labels = save['test_dataset'],save['test_labels']

        train_dataset = train_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
        valid_dataset = valid_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)
        test_dataset = test_dataset.reshape((-1,image_size,image_size,num_channels)).astype(np.float32)

        print('\tFinal shape (train):%s',train_dataset.shape)
        print('\tFinal shape (valid):%s',valid_dataset.shape)
        print('\tFinal shape (test):%s',test_dataset.shape)

        train_labels = (np.arange(num_labels) == train_labels[:,None]).astype(np.float32)
        valid_labels = (np.arange(num_labels) == valid_labels[:,None]).astype(np.float32)
        test_labels = (np.arange(num_labels) == test_labels[:,None]).astype(np.float32)

        print('\tFinal shape (train) labels:%s',train_labels.shape)
        print('\tFinal shape (valid) labels:%s',valid_labels.shape)
        print('\tFinal shape (test) labels:%s',test_labels.shape)

        train_size,valid_size,test_size = train_dataset.shape[0],valid_dataset.shape[0],test_dataset.shape[0]

def load_data_notMNIST():
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


def reshape_data_notMNIST():
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
                                    stddev=2./(hyparams[op]['weights'][0]*hyparams[op]['weights'][1])
                                    )
            )
            biases[op] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[hyparams[op]['weights'][3]]))
        if 'incept' in op:
            print('\n\tDefining the weights and biases for the Incept Module')
            inc_hyparams = hyparams[op]
            for k,v in inc_hyparams.items():
                if 'conv' in k:
                    w_key = op+'_'+k
                    print('\t\tParameters for %s'%w_key)
                    print('\t\t\tWeights:%s'%inc_hyparams[k]['weights'])
                    print('\t\t\tBias:%d'%inc_hyparams[k]['weights'][3])
                    weights[w_key] = tf.Variable(
                        tf.truncated_normal(inc_hyparams[k]['weights'],
                                            stddev=2./(inc_hyparams[k]['weights'][0] *
                                                       inc_hyparams[k]['weights'][1] *
                                                       inc_hyparams[k]['weights'][2])
                                            )
                    )
                    biases[w_key] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[inc_hyparams[k]['weights'][3]]))

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

                biases[op] = tf.Variable(tf.constant(np.random.random()*0.01,shape=[hyparams[op]['out']]))


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
        if 'incept' in op:
            print('\tInception for data ...')

            tf_incept_out = None

            conv_1x1_id = op + '_' + 'iconv_1x1'
            for inc_op in incept_orders[op]:
                inc_op_id = op + '_' + inc_op

                if 'pool' in inc_op:
                    print('\t\tPooling %s'%inc_op_id)
                    # pooling followed by 1x1 convolution
                    tmp_x = tf.nn.avg_pool(x,
                                           ksize=hyparams[op][inc_op]['kernel'],
                                           strides=hyparams[op][inc_op]['stride'],
                                           padding=hyparams[op][inc_op]['padding']
                                           )

                    #1x1 convolution with iconv_1x1

                    tmp_x = tf.nn.conv2d(
                        tmp_x,weights[conv_1x1_id],
                        hyparams[op]['iconv_1x1']['stride'],
                        padding=hyparams[op]['iconv_1x1']['padding']
                    )
                    # relu activation
                    tmp_x = tf.nn.relu(tmp_x + biases[conv_1x1_id])

                    if tf_incept_out is None:
                        tf_incept_out = tf.identity(tmp_x)
                    else:
                        tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])

                    print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))
                if 'conv' in inc_op:
                    print('\t\tConvolving %s'%inc_op_id)

                    # no following convolution after 1x1 convolution
                    if inc_op=='iconv_1x1':
                        tmp_x = tf.nn.conv2d(x,
                                             weights[conv_1x1_id],
                                             hyparams[op]['iconv_1x1']['stride'],
                                             padding=hyparams[op]['iconv_1x1']['padding']
                                             )
                        #relu activation
                        tmp_x = tf.nn.relu(tmp_x + biases[conv_1x1_id])

                        if tf_incept_out is None:
                            tf_incept_out = tf.identity(tmp_x)
                        else:
                            tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])
                        print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))

                    else:
                        # 1x1 convolution
                        tmp_x = tf.nn.conv2d(x,
                                             weights[conv_1x1_id],
                                             hyparams[op]['iconv_1x1']['stride'],
                                             padding=hyparams[op]['iconv_1x1']['padding']
                                             )
                        #relu activation
                        tmp_x = tf.nn.relu(tmp_x + biases[conv_1x1_id])

                        #5x5 or 3x3 convolution
                        tmp_x = tf.nn.conv2d(tmp_x,
                                             weights[inc_op_id],
                                             hyparams[op][inc_op]['stride'],
                                             padding=hyparams[op][inc_op]['padding']
                        )
                        #relu activation
                        tmp_x = tf.nn.relu(tmp_x + biases[inc_op_id])

                        if tf_incept_out is None:
                            tf_incept_out = tf.identity(tmp_x)
                        else:
                            tf_incept_out = tf.concat(3,[tf_incept_out,tmp_x])
                        print('\n\t\tStacked input of Inception module after %s, %s'%(inc_op,tf_incept_out.get_shape().as_list()))

            print('\n\t\tFinal stacked input of Inception module, %s'%tf_incept_out.get_shape().as_list())
            x=tf_incept_out

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
    if include_l2_loss:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels)) + \
               (beta/2)*tf.reduce_sum([tf.nn.l2_loss(w) if 'fulcon' in kw or 'cov' in kw else 0 for kw,w in weights.items()])
    else:
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, labels))

    return loss

def optimize_func(loss,global_step):
    # Optimizer.
    if decay_learning_rate:
        learning_rate = tf.train.exponential_decay(start_lr, global_step,decay_steps=500,decay_rate=0.99)
    else:
        learning_rate = start_lr

    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    return optimizer,learning_rate

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
    global train_size,valid_size,test_size

    #load_data_notMNIST()
    load_data_cifar10()
    #reshape_data_notMNIST()
    reformat_data_cifar10()
    graph = tf.Graph()

    valid_accuracies = []

    with tf.Session(graph=graph) as session:
        #tf.global_variables_initializer().run()
        # Input data.
        global_step = tf.Variable(0, trainable=False)

        print('Input data defined...\n')
        tf_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_size, image_size, num_channels))
        tf_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_valid_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
        tf_valid_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        tf_test_dataset = tf.placeholder(tf.float32, shape=(batch_size,image_size,image_size,num_channels))
        tf_test_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))

        create_subsample_layers()

        print('================ Training ==================\n')
        logits = get_logits(tf_dataset)
        loss = calc_loss(logits,tf_labels)
        pred = predict_with_logits(logits)
        optimize = optimize_func(loss,global_step)
        inc_gstep = inc_global_step(global_step)
        print('==============================================\n')

        print('================ Validating ==================\n')
        valid_pred = predict_with_dataset(tf_valid_dataset)
        print('==============================================\n')
        print('================ Testing ==================\n')
        test_pred = predict_with_dataset(tf_test_dataset)
        print('==============================================\n')

        tf.initialize_all_variables().run()
        print('Initialized...')
        print('Batch size:',batch_size)
        print('Depths: ',depth_conv)
        print('Num Steps: ',num_steps)
        print('Decay Learning Rate: ',decay_learning_rate,', ',start_lr)
        print('Dropout: ',use_dropout,', ',dropout_rate)
        print('Early Stopping: ',early_stopping)
        print('Include L2, Beta: ',include_l2_loss,', ',beta)
        print('==================================================\n')

        accuracy_drop = 0 # used for early stopping
        for step in range(num_steps):
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
            batch_labels = train_labels[offset:(offset + batch_size), :]

            feed_dict = {tf_dataset : batch_data, tf_labels : batch_labels}
            _, l, (_,updated_lr), predictions,_ = session.run([logits,loss,optimize,pred,inc_gstep], feed_dict=feed_dict)

            if step % 50 == 0:
                print('Global step: %d'%global_step.eval())
                print('Minibatch loss at step %d: %f' % (step, l))
                print('Learning rate: %.3f'%updated_lr)
                print('Minibatch accuracy: %.1f%%' % accuracy(predictions, batch_labels))
                v_acc_arr = None

                for batch_id in range(valid_size//batch_size):
                    batch_valid_data = valid_dataset[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:]
                    batch_valid_labels = valid_labels[batch_id*batch_size:(batch_id+1)*batch_size,:]

                    feed_valid_dict = {tf_valid_dataset:batch_valid_data, tf_valid_labels:batch_valid_labels}
                    valid_predictions = session.run([valid_pred],feed_dict=feed_valid_dict)
                    if v_acc_arr is None:
                        v_acc_arr = np.asarray(valid_predictions[0],dtype=np.float32)
                    else:
                        v_acc_arr = np.append(v_acc_arr,valid_predictions[0],axis=0)

                v_accuracy = accuracy(v_acc_arr, valid_labels)
                print('Validation accuracy: %.1f%%' %v_accuracy)

                if early_stopping and step>2500 and len(valid_accuracies)>0:
                    # the accuracy drop needs to happen consecutively else we reset accuracy drop
                    if v_accuracy < np.mean(valid_accuracies[-10:-1])*0.95:
                        accuracy_drop += 1
                    else:
                        accuracy_drop = 0

                if accuracy_drop>accuracy_drops_cap:
                    print('Accuracy drop exceeded the threshold. Halting the training')
                    break

                valid_accuracies.append(v_accuracy)

        ts_acc_arr = None
        for batch_id in range(valid_size//batch_size):
            batch_test_data = test_dataset[batch_id*batch_size:(batch_id+1)*batch_size,:,:,:]
            batch_test_labels = test_labels[batch_id*batch_size:(batch_id+1)*batch_size,:]

            feed_test_dict = {tf_test_dataset:batch_test_data, tf_test_labels:batch_test_labels}
            test_predictions = session.run([test_pred],feed_dict=feed_test_dict)

            if ts_acc_arr is None:
                ts_acc_arr = np.asarray(test_predictions[0],dtype=np.float32)
            else:
                ts_acc_arr = np.append(ts_acc_arr,test_predictions[0],axis=0)

        print('Test accuracy: %.1f%%' % accuracy(ts_acc_arr, test_labels))