# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet.gluon import nn

class gcn_layer(nn.HybridBlock):
    def __init__(self, num_of_filters, **kwargs):
        super(gcn_layer, self).__init__(**kwargs)
        with self.name_scope():
            self.fc = nn.Dense(num_of_filters)
        
    def hybrid_forward(self, F, x, A_):
        '''
        Parameters
        ----------
        A_, D^{-1/2} A D^{-1/2}
        '''
        return self.fc(F.dot(A_, x))
        
class GCN(nn.HybridBlock):
    def __init__(self, **kwargs):
        super(GCN, self).__init__(**kwargs)
        with self.name_scope():
            self.gcn1 = gcn_layer(256)
            self.gcn2 = gcn_layer(7)
        
    def hybrid_forward(self, F, x, A_):
        return self.gcn2(F.relu(self.gcn1(x, A_)), A_)