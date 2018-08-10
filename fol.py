"""

First Order Logic (FOL) rules

"""

import warnings
import numpy
import theano.tensor.shared_randomstreams
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.tensor.signal import pool
from theano.tensor.nnet import conv
from theano import printing


class FOL(object):
    """ First Order Logic (FOL) rules """
    
    def __init__(self, K, input, fea):
        """ Initialize
    
    : type K: int
    : param K: the number of classes 
    """
        self.input = input
        self.fea = fea
        # Record the data relevance (binary)
        self.conds = self.conditions(self.input, self.fea)
        self.K = K

    def conditions(self, X, F):
        results,_ = theano.scan(lambda x,f: self.condition_single(x,f), sequences=[X,F])
        return results
        

    def distribution_helper_helper(self, x, f):
        results,_ = theano.scan(lambda k: self.value_single(x,k,f), sequences=T.arange(self.K))
        return results


    def distribution_helper(self, w, X, F, conds):
        nx = X.shape[0]
        distr = T.alloc(1.0, nx, self.K)
        distr,_ = theano.scan( 
            lambda c,x,f,d: ifelse(T.eq(c,1.), self.distribution_helper_helper(x,f), d),
            sequences=[conds, X, F, distr])
        distr,_ = theano.scan(
            lambda d: -w*(T.min(d,keepdims=True)-d), # relative value w.r.t the minimum
            sequences=distr)
        return distr


    """
    Interface function of logic constraints

    The interface is general---only need to overload condition_single(.) and
    value_single(.) below to implement a logic rule---but can be slow

    See the overloaded log_distribution(.) of the BUT-rule for an efficient
    version specific to the BUT-rule
    """
    def log_distribution(self, w, X=None, F=None, config={}):
        """ Return an nxK matrix with the (i,c)-th term
    = - w * (1 - r(X_i, y_i=c))
           if X_i is a grounding of the rule
    = 1    otherwise
    """
        if F == None:
            X, F, conds = self.input, self.fea, self.conds
        else:
            conds = self.conditions(X,F)
        log_distr = self.distribution_helper(w,X,F,conds)
        return log_distr


    """
    Rule-specific functions to be overloaded    

    """
    def condition_single(self, x, f):
        """ True if x satisfies the condition """
        return T.cast(0, dtype=theano.config.floatX)


    def value_single(self, x, y, f):
        """ value = r(x,y) """
        return T.cast(1, dtype=theano.config.floatX)


#----------------------------------------------------
# BUT rule
#----------------------------------------------------

class FOL_But(FOL):
    """ x=x1_but_x2 => { y => pred(x2) AND pred(x2) => y } """
    def __init__(self, K, input, fea):
        """ Initialize
    
    :type K: int
    :param K: the number of classes 

    :type fea: theano.tensor.dtensor4
    :param fea: symbolic feature tensor, of shape 3
                fea[0]   : 1 if x=x1_but_x2, 0 otherwise
                fea[1:2] : classifier.predict_p(x_2)
    """
        assert K == 2
        super(FOL_But, self).__init__(K, input, fea)

    """
    Rule-specific functions

    """
    def condition_single(self, x, f):
        return T.cast(T.eq(f[0],1.), dtype=theano.config.floatX)
        

    def value_single(self, x, y, f):
        ret = T.mean([T.min([1.-y+f[2],1.]), T.min([1.-f[2]+y,1.])])
        ret = T.cast(ret, dtype=theano.config.floatX)
        return T.cast(ifelse(T.eq(self.condition_single(x,f),1.), ret, 1.),
                      dtype=theano.config.floatX)

    """
    Efficient version specific to the BUT-rule

    """
    def log_distribution(self, w, X=None, F=None):
        if F == None:
            X, F = self.input, self.fea
        F_mask = F[:,0] 
        F_fea = F[:,1:]
        # y = 0
        distr_y0 = w*F_mask*F_fea[:,0]
        # y = 1 
        distr_y1 = w*F_mask*F_fea[:,1]
        distr_y0 = distr_y0.reshape([distr_y0.shape[0],1])
        distr_y1 = distr_y1.reshape([distr_y1.shape[0],1])
        distr = T.concatenate([distr_y0, distr_y1], axis=1)
        return distr



