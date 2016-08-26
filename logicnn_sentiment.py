import cPickle
import numpy as np
from collections import defaultdict, OrderedDict
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import re
import warnings
import sys
import time
import os
import math

warnings.filterwarnings("ignore")   
execfile("logicnn_classes.py") 
       
def train_conv_net(datasets,
                   U,
                   word_idx_map,
                   img_w=300, 
                   filter_hs=[3,4,5],
                   hidden_units=[100,2], 
                   dropout_rate=[0.5],
                   shuffle_batch=True,
                   n_epochs=25, 
                   batch_size=50, 
                   lr_decay = 0.95,
                   conv_non_linear="relu",
                   activations=[Iden],
                   sqr_norm_lim=9,
                   non_static=True,
                   pi_params=[1.,0],
                   C=1.0): 
    """
    Train a convnet through iterative distillation
    img_h = sentence length (padded where necessary)
    img_w = word vector length (300 for word2vec)
    filter_hs = filter window sizes    
    hidden_units = [x,y] x is the number of feature maps (per filter window), and y is the penultimate layer
    sqr_norm_lim = s^2 in the paper [Kim, 2014]
    lr_decay = adadelta decay parameter
    pi_params = update strategy of imitation parameter \pi
    C = regularization strength
    """    
    rng = np.random.RandomState(3435)
    img_h = len(datasets[0][0])-1  
    filter_w = img_w    
    feature_maps = hidden_units[0]
    filter_shapes = []
    pool_sizes = []
    for filter_h in filter_hs:
        filter_shapes.append((feature_maps, 1, filter_h, filter_w))
        pool_sizes.append((img_h-filter_h+1, img_w-filter_w+1))
    parameters = [("image shape",img_h,img_w),("filter shape",filter_shapes), ("hidden_units",hidden_units),
                  ("dropout", dropout_rate), ("batch_size",batch_size),("non_static", non_static),
                    ("learn_decay",lr_decay), ("conv_non_linear", conv_non_linear), ("non_static", non_static),
                    ("sqr_norm_lim",sqr_norm_lim),("shuffle_batch",shuffle_batch),("pi_params",pi_params),("C",C)]
    print parameters 
    
    #define model architecture
    index = T.lscalar()
    x = T.matrix('x')   
    y = T.ivector('y')
    Words = theano.shared(value = U, name = "Words")
    zero_vec_tensor = T.vector()
    zero_vec = np.zeros(img_w)
    set_zero = theano.function([zero_vec_tensor], updates=[(Words, T.set_subtensor(Words[0,:], zero_vec_tensor))], allow_input_downcast=True)
    layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((x.shape[0],1,x.shape[1],Words.shape[1]))                                  
    conv_layers = []
    layer1_inputs = []
    for i in xrange(len(filter_hs)):
        filter_shape = filter_shapes[i]
        pool_size = pool_sizes[i]
        conv_layer = LeNetConvPoolLayer(rng, input=layer0_input,image_shape=(batch_size, 1, img_h, img_w),
                                filter_shape=filter_shape, poolsize=pool_size, non_linear=conv_non_linear)
        layer1_input = conv_layer.output.flatten(2)
        conv_layers.append(conv_layer)
        layer1_inputs.append(layer1_input)
    layer1_input = T.concatenate(layer1_inputs,1)
    hidden_units[0] = feature_maps*len(filter_hs)    
    classifier = MLPDropout(rng, input=layer1_input, layer_sizes=hidden_units, activations=activations, dropout_rates=dropout_rate)
  
    #build the feature of BUT-rule
    f_but = T.fmatrix('f_but')
    f_but_ind = T.fmatrix('f_ind') # indicators
    f_but_layer0_input = Words[T.cast(f_but.flatten(),dtype="int32")].reshape((f_but.shape[0],1,f_but.shape[1],Words.shape[1]))
    f_but_pred_layers = []
    for conv_layer in conv_layers:
        f_but_layer0_output = conv_layer.predict(f_but_layer0_input, batch_size)
        f_but_pred_layers.append(f_but_layer0_output.flatten(2))
    f_but_layer1_input = T.concatenate(f_but_pred_layers, 1)
    f_but_y_pred_p = classifier.predict_p(f_but_layer1_input)
    f_but_full = T.concatenate([f_but_ind,f_but_y_pred_p],axis=1) # batch_size x 1 + batch_size x K
    f_but_full = theano.gradient.disconnected_grad(f_but_full)

    #add logic layer
    nclasses = 2
    rules = [FOL_But(nclasses, x, f_but_full)]
    rule_lambda = [1]
    print 'rule_lambda ', rule_lambda
    new_pi = get_pi(cur_iter=0, params=pi_params)
    logic_nn = LogicNN(rng, input=x, network=classifier, rules=rules, rule_lambda=rule_lambda, pi=new_pi, C=C)
 
    #define parameters of the model and update functions using adadelta
    params_p = logic_nn.params_p
    for conv_layer in conv_layers:
        params_p += conv_layer.params
    if non_static:
        #if word vectors are allowed to change, add them as model parameters
        params_p += [Words]
    cost_p = logic_nn.negative_log_likelihood(y) 
    dropout_cost_p = logic_nn.dropout_negative_log_likelihood(y) 
    grad_updates_p = sgd_updates_adadelta(params_p, dropout_cost_p, lr_decay, 1e-6, sqr_norm_lim)
    
    #shuffle dataset and assign to mini batches. if dataset size is not a multiple of mini batches, replicate 
    #extra data (at random)
    np.random.seed(3435)
    if datasets[0].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[0].shape[0] % batch_size
        # shuffle both train data and features
        permutation_order = np.random.permutation(datasets[0].shape[0])
        #
        train_set = datasets[0][permutation_order]
        extra_data = train_set[:extra_data_num]
        new_data=np.append(datasets[0],extra_data,axis=0)
        #
        new_fea = {}
        train_fea = datasets[3]
        for k in train_fea.keys():
            train_fea_k = train_fea[k][permutation_order]
            extra_fea = train_fea_k[:extra_data_num]
            new_fea[k] = np.append(train_fea[k],extra_fea,axis=0)
        #
        train_text = datasets[6][permutation_order]
        extra_text = train_text[:extra_data_num]
        new_text=np.append(datasets[6],extra_text,axis=0)
    else:
        new_data = datasets[0]
        new_fea = datasets[3]
        new_text = datasets[6]
    # shuffle both train data and features
    permutation_order = np.random.permutation(new_data.shape[0])
    new_data = new_data[permutation_order]
    for k in new_fea.keys():
        new_fea[k] = new_fea[k][permutation_order]
    new_text = new_text[permutation_order]
    n_batches = new_data.shape[0]/batch_size
    n_train_batches = n_batches
    train_set = new_data
    train_set_x, train_set_y = shared_dataset((train_set[:,:img_h],train_set[:,-1]))
    train_fea = new_fea
    train_fea_but_ind = train_fea['but_ind'].reshape([train_fea['but_ind'].shape[0],1])
    train_fea_but_ind = shared_fea(train_fea_but_ind)
    for k in new_fea.keys():
        if k!='but_text':
            train_fea[k] = shared_fea(new_fea[k])

    if datasets[1].shape[0] % batch_size > 0:
        extra_data_num = batch_size - datasets[1].shape[0] % batch_size
        # shuffle both val data and features
        permutation_order = np.random.permutation(datasets[1].shape[0])
        #
        val_set = datasets[1][permutation_order]
        extra_data = val_set[:extra_data_num]
        new_val_data=np.append(datasets[1],extra_data,axis=0)
        #
        new_val_fea = {}
        val_fea = datasets[4]
        for k in val_fea.keys():
            val_fea_k = val_fea[k][permutation_order]
            extra_fea = val_fea_k[:extra_data_num]
            new_val_fea[k] = np.append(val_fea[k],extra_fea,axis=0)
        #
        val_text = datasets[7][permutation_order]
        extra_text = val_text[:extra_data_num]
        new_val_text = np.append(datasets[7],extra_text,axis=0)
    else:
        new_val_data = datasets[1]
        new_val_fea = datasets[4]
        new_val_text = datasets[7]
    val_set = new_val_data
    val_set_x, val_set_y = shared_dataset((val_set[:,:img_h],val_set[:,-1]))
    n_batches = new_val_data.shape[0] / batch_size
    n_val_batches = n_batches
    val_fea = new_val_fea
    val_fea_but_ind = val_fea['but_ind'].reshape([val_fea['but_ind'].shape[0],1])
    val_fea_but_ind = shared_fea(val_fea_but_ind)
    for k in val_fea.keys():
        if k!='but_text':
            val_fea[k] = shared_fea(val_fea[k])

    test_set_x = datasets[2][:,:img_h] 
    test_set_y = np.asarray(datasets[2][:,-1],"int32")
    test_fea = datasets[5]
    test_fea_but_ind = test_fea['but_ind']
    test_fea_but_ind = test_fea_but_ind.reshape([test_fea_but_ind.shape[0],1])
    test_text = datasets[8]

    ### compile theano functions to get train/val/test errors
    val_model = theano.function([index], logic_nn.errors(y),
         givens={
            x: val_set_x[index * batch_size: (index + 1) * batch_size],
             y: val_set_y[index * batch_size: (index + 1) * batch_size],
             f_but: val_fea['but'][index * batch_size: (index + 1) * batch_size],
             f_but_ind: val_fea_but_ind[index * batch_size: (index + 1) * batch_size,:]},
                                allow_input_downcast=True,
                                on_unused_input='warn')
            
    test_model = theano.function([index], logic_nn.errors(y),
             givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                 y: train_set_y[index * batch_size: (index + 1) * batch_size],
                 f_but: train_fea['but'][index * batch_size: (index + 1) * batch_size],
                 f_but_ind: train_fea_but_ind[index * batch_size: (index + 1) * batch_size,:]},
                                 allow_input_downcast=True,
                                 on_unused_input='warn')

    train_model = theano.function([index], cost_p, updates=grad_updates_p,
          givens={
            x: train_set_x[index*batch_size:(index+1)*batch_size],
              y: train_set_y[index*batch_size:(index+1)*batch_size],
              f_but: train_fea['but'][index * batch_size: (index + 1) * batch_size],
              f_but_ind: train_fea_but_ind[index * batch_size: (index + 1) * batch_size,:]},
                                  allow_input_downcast = True,
                                  on_unused_input='warn')

    ### setup testing
    test_size = test_set_x.shape[0]
    print 'test size ', test_size 
    test_pred_layers = []
    test_layer0_input = Words[T.cast(x.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    f_but_test_pred_layers = []
    f_but_test_layer0_input = Words[T.cast(f_but.flatten(),dtype="int32")].reshape((test_size,1,img_h,Words.shape[1]))
    for conv_layer in conv_layers:
        test_layer0_output = conv_layer.predict(test_layer0_input, test_size)
        test_pred_layers.append(test_layer0_output.flatten(2))
        f_but_test_layer0_output = conv_layer.predict(f_but_test_layer0_input, test_size)
        f_but_test_pred_layers.append(f_but_test_layer0_output.flatten(2))
    test_layer1_input = T.concatenate(test_pred_layers, 1)
    f_but_test_layer1_input = T.concatenate(f_but_test_pred_layers, 1)
    f_but_test_y_pred_p = classifier.predict_p(f_but_test_layer1_input)
    f_but_test_full = T.concatenate([f_but_ind,f_but_test_y_pred_p],axis=1) # Ns x 1 + Ns x K

    # transform to shared variables
    test_set_x_shr, test_set_y_shr = shared_dataset((test_set_x,test_set_y))

    test_q_y_pred, test_p_y_pred = logic_nn.predict(test_layer1_input,
                                       test_set_x_shr,
                                       [f_but_test_full])
    test_q_error = T.mean(T.neq(test_q_y_pred, y))
    test_p_error = T.mean(T.neq(test_p_y_pred, y))
    test_model_all = theano.function([x,y,f_but,f_but_ind],
                                     [test_q_error, test_p_error], allow_input_downcast = True,
                                     on_unused_input='warn')
   
    ### start training over mini-batches
    print '... training'
    epoch = 0
    batch = 0
    cost_epoch = 0    
    while (epoch < n_epochs):
        start_time = time.time()
        epoch = epoch + 1
        # train
        if shuffle_batch:
            for minibatch_index in np.random.permutation(range(n_train_batches)):
                batch = batch + 1
                new_pi = get_pi(cur_iter=batch*1./n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)
                cost_epoch = train_model(minibatch_index)
                set_zero(zero_vec)
        else:
            for minibatch_index in xrange(n_train_batches):
                batch = batch + 1
                new_pi = get_pi(cur_iter=batch*1./n_train_batches, params=pi_params)
                logic_nn.set_pi(new_pi)
                cost_epoch = train_model(minibatch_index)  
                set_zero(zero_vec)
        # eval
        train_losses = [test_model(i) for i in xrange(n_train_batches)]
        train_losses = np.array(train_losses)
        train_q_perf = 1 - np.mean(train_losses[:,0])
        train_p_perf = 1 - np.mean(train_losses[:,1])
        print('epoch: %i, training time: %.2f secs; (q): train perf: %.4f %%; (p): train perf: %.4f %%' % \
               (epoch, time.time()-start_time, train_q_perf * 100., train_p_perf * 100.))
        test_loss = test_model_all(test_set_x,test_set_y,test_fea['but'],test_fea_but_ind)
        test_loss = np.array(test_loss)
        test_perf = 1 - test_loss
        print 'test perf: q %.4f %%, p %.4f %%' % (test_perf[0]*100., test_perf[1]*100.)
    return test_perf


def get_pi(cur_iter, params=None, pi=None):
    """ exponential decay: pi_t = max{1 - k^t, lb} """
    k,lb = params[0],params[1]
    pi = 1. - max([k**cur_iter, lb])
    return pi


def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                               dtype=theano.config.floatX),
                                 borrow=borrow)
        return shared_x, T.cast(shared_y, 'int32')
        
def shared_fea(fea, borrow=True):
    """ 
    Function that loads the fea into shared variables
    """
    shared_fea = theano.shared(np.asarray(fea,
                                          dtype=theano.config.floatX),
                             borrow=borrow)
    return shared_fea 


def sgd_updates_adadelta(params,cost,rho=0.95,epsilon=1e-6,norm_lim=9,word_vec_name='Words'):
    """
    adadelta update rule, mostly from
    https://groups.google.com/forum/#!topic/pylearn-dev/3QbKtCumAW4 (for Adadelta)
    """
    updates = OrderedDict({})
    exp_sqr_grads = OrderedDict({})
    exp_sqr_ups = OrderedDict({})
    gparams = []
    for param in params:
        empty = np.zeros_like(param.get_value())
        exp_sqr_grads[param] = theano.shared(value=as_floatX(empty),name="exp_grad_%s" % param.name)
        gp = T.grad(cost, param)
        exp_sqr_ups[param] = theano.shared(value=as_floatX(empty), name="exp_grad_%s" % param.name)
        gparams.append(gp)
    for param, gp in zip(params, gparams):
        exp_sg = exp_sqr_grads[param]
        exp_su = exp_sqr_ups[param]
        up_exp_sg = rho * exp_sg + (1 - rho) * T.sqr(gp)
        updates[exp_sg] = up_exp_sg
        step =  -(T.sqrt(exp_su + epsilon) / T.sqrt(up_exp_sg + epsilon)) * gp
        updates[exp_su] = rho * exp_su + (1 - rho) * T.sqr(step)
        if param.name=='Words':
            stepped_param = param + step * .5
        else:
            stepped_param = param + step
        if (param.get_value(borrow=True).ndim == 2) and (param.name!='Words'):
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, T.sqrt(norm_lim))
            scale = desired_norms / (1e-7 + col_norms)
            updates[param] = stepped_param * scale
        else:
            updates[param] = stepped_param      
    return updates

def as_floatX(variable):
    if isinstance(variable, float):
        return np.cast[theano.config.floatX](variable)

    if isinstance(variable, np.ndarray):
        return np.cast[theano.config.floatX](variable)
    return theano.tensor.cast(variable, theano.config.floatX)
    
def safe_update(dict_to, dict_from):
    """
    re-make update dictionary for safe updating
    """
    for key, val in dict(dict_from).iteritems():
        if key in dict_to:
            raise KeyError(key)
        dict_to[key] = val
    return dict_to
    

def get_idx_from_sent(sent, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentence into a list of indices. Pad with zeroes.
    """
    x = []
    pad = filter_h - 1
    for i in xrange(pad):
        x.append(0)
    words = sent.split()
    for word in words:
        if word in word_idx_map:
            x.append(word_idx_map[word])
    while len(x) < max_l+2*pad:
        x.append(0)
    return x

def get_idx_from_but_fea(but_fea, but_ind, word_idx_map, max_l=51, k=300, filter_h=5):
    if but_ind == 0:
        pad = filter_h - 1
        x = [0]*(max_l+2*pad)
    else:
        x = get_idx_from_sent(but_fea, word_idx_map, max_l, k, filter_h)
    return x

def make_idx_data(revs, fea, word_idx_map, max_l=51, k=300, filter_h=5):
    """
    Transforms sentences into a 2-d matrix.
    """
    train, dev, test = [], [], []
    train_text, dev_text, test_text = [], [], []
    train_fea, dev_fea, test_fea = {}, {}, {}
    fea['but'] = []
    for k in fea.keys():
        train_fea[k], dev_fea[k], test_fea[k] = [],[],[]
    for i,rev in enumerate(revs):
        sent = get_idx_from_sent(rev["text"], word_idx_map, max_l, k, filter_h)
        sent.append(rev["y"])
        fea['but'].append(get_idx_from_but_fea(fea['but_text'][i], fea['but_ind'][i], word_idx_map, max_l, k, filter_h))
        if rev["split"]==0:
            train.append(sent)
            for k,v in fea.iteritems():
                train_fea[k].append(v[i])
            train_text.append(rev["text"])
        elif rev["split"]==1:
            dev.append(sent)
            for k,v in fea.iteritems():
                dev_fea[k].append(v[i])
            dev_text.append(rev["text"])
        else:  
            test.append(sent)
            for k,v in fea.iteritems():
                test_fea[k].append(v[i])
            test_text.append(rev["text"])
    train = np.array(train,dtype="int")
    dev = np.array(dev,dtype="int")
    test = np.array(test,dtype="int")
    for k in fea.keys():
        if k=='but':
            train_fea[k] = np.array(train_fea[k],dtype='int')
            dev_fea[k] = np.array(dev_fea[k],dtype='int')
            test_fea[k] = np.array(test_fea[k],dtype='int')
        elif k=='but_text':
            train_fea[k] = np.array(train_fea[k])
            dev_fea[k] = np.array(dev_fea[k])
            test_fea[k] = np.array(test_fea[k])
        else:
            train_fea[k] = np.array(train_fea[k],dtype=theano.config.floatX)
            dev_fea[k] = np.array(dev_fea[k],dtype=theano.config.floatX)
            test_fea[k] = np.array(test_fea[k],dtype=theano.config.floatX)
    train_text = np.array(train_text)
    dev_text = np.array(dev_text)
    test_text = np.array(test_text)
    return [train, dev, test, train_fea, dev_fea, test_fea, train_text, dev_text, test_text]
  
   
if __name__=="__main__":
    print "loading data...",
    #path = 'data/' 
    path = '../release_bak/data/sst/' 
    print path
    x = cPickle.load(open("%s/stsa.binary.p"%path,"rb"))
    revs, W, W2, word_idx_map, vocab = x[0], x[1], x[2], x[3], x[4]
    print "data loaded!"
    print "loading features..."
    fea = cPickle.load(open("%s/stsa.binary.p.fea.p"%path,"rb"))
    print "features loaded!"

    mode= sys.argv[1]
    word_vectors = sys.argv[2] 
    if mode=="-nonstatic":
        print "model architecture: CNN-non-static"
        non_static=True
    elif mode=="-static":
        print "model architecture: CNN-static"
        non_static=False
    if word_vectors=="-rand":
        print "using: random vectors"
        U = W2
    elif word_vectors=="-word2vec":
        print "using: word2vec vectors, dim=%d" % W.shape[1]
        U = W    

    # q: teacher network; p: student network
    q_results = []
    p_results = []
    datasets = make_idx_data(revs, fea, word_idx_map, max_l=53, k=300, filter_h=5)
    perf = train_conv_net(datasets,
                          U,
                          word_idx_map,
                          img_w=W.shape[1],
                          lr_decay=0.95,
                          filter_hs=[3,4,5],
                          conv_non_linear="relu",
                          hidden_units=[100,2], 
                          shuffle_batch=True, 
                          n_epochs=11,
                          sqr_norm_lim=9,
                          non_static=non_static,
                          batch_size=50,
                          dropout_rate=[0.4],
                          pi_params=[0.95,0],
                          C=6.)
    q_results.append(perf[0])
    p_results.append(perf[1])
    print 'teacher network q: ',str(np.mean(q_results))
    print 'studnet network p: ',str(np.mean(p_results))
