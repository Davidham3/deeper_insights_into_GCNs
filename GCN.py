# -*- coding:utf-8 -*-

import mxnet as mx
from mxnet import nd
from mxnet import autograd
from mxnet import gluon

import numpy as np

from sklearn.metrics import accuracy_score

from utils import *
from model import GCN

# change ctx to mx.gpu(0) to use gpu device
ctx = mx.cpu()

if __name__ == "__main__":
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data('cora')
    features = nd.array(features.toarray(), ctx = ctx)
    y_train = nd.array(y_train, ctx = ctx)
    y_val = nd.array(y_val, ctx = ctx)

    A_tilde = adj.toarray() + np.identity(adj.shape[0])
    D = A_tilde.sum(axis = 1)
    A_ = nd.array(np.diag(D ** -0.5).dot(A_tilde).dot(np.diag(D ** -0.5)), ctx = ctx)

    idx = np.arange(len(A_))

    net = GCN()
    net.initialize(ctx = ctx)
    net.hybridize()
    output = net(features, A_)

    loss_function = gluon.loss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': 1e-3})

    for epoch in range(100):
        with autograd.record():
            output = net(features, A_)
            l = loss_function(output[idx[train_mask]], nd.argmax(y_train[idx[train_mask]], axis = 1))
        l.backward()
        trainer.step(1)
        print('training loss: %.2f'%(l.mean().asnumpy()[0]))
        
        output = net(features, A_)
        l = loss_function(output[idx[val_mask]], nd.argmax(y_val[idx[val_mask]], axis = 1))
        print('validation loss %.2f'%(l.mean().asnumpy()[0]))
        print()
    
    output = net(features, A_)
    print('testing accuracy: %.3f'%(accuracy_score(np.argmax(y_test[idx[test_mask]], axis = 1), nd.argmax(output[idx[test_mask]], axis = 1).asnumpy())))
    