import os
import sys
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from config import cfg
from utils import load_data
from cnet import CapsNet

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
n_itrs = 1000
batch_size = 128
num_label = 10
model = CapsNet()

print("session started")
for step in range(n_itrs):
    
    batch_xs, batch_ys = mnist.train.next_batch(cfg.batch_size)
    batch_xs = batch_xs.reshape(cfg.batch_size, 28, 28,1)
    batch_ys = batch_ys.reshape(cfg.batch_size, 10)
    loss, acc =  model.train(batch_xs,batch_ys)
    assert not np.isnan(loss), 'Something wrong! loss is nan...'
    if (step % 50 == 0):
        print("step-{} summary: loss= {}, training accuracy = {}".format(step, loss,acc))

pred = model.predict(batch_xs)
print("training accuracy=",acc)


print("Testing started")
n_steps = 100
corr_pred = 0
for step in range(n_steps):
    print("---- {} ----".format(step))
    batch_xs, batch_ys = mnist.test.next_batch(cfg.batch_size)
    batch_xs = batch_xs.reshape(cfg.batch_size, 28, 28,1)
    batch_ys = batch_ys.reshape(cfg.batch_size, 10)
    
    pred = model.predict(batch_xs)
    corr_pred = corr_pred + np.sum(np.argmax(batch_ys,axis =1) == pred)
    if (step % 20 == 0):
        print("testing accuracy", corr_pred)

print("testing accuracy:",corr_pred/(n_steps * cfg.batch_size))

print("---- {} ----".format(step))
