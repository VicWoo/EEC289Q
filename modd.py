#This module is to pass the extracted features through different combinations of pre-trained dnn models imported from both Gluon model zoo and keras library
#We are using transfer learning in order to simplify the training process and thus shorten the amount of time required for running each training test

import mxnet as mx
from mxnet import init, gluon, nd, autograd, image
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os
from glob import glob
import matplotlib.pyplot as plt
from mxnet import image


#Change the following to mx.cpu() if you don't have GPU in your computer.
#To use different GPU, you can try "ctx = mx.gpu(1)", where 1 is the first GPU.
ctx = mx.gpu()
data_dir        = 'data'
batch_size      = 128
learning_rate   = 1e-4
submit_fileName = 'pred.csv'

synset = list(pd.read_csv(os.path.join('.', data_dir, 'sample_submission.csv')).columns[1:])
n = len(glob(os.path.join('.', data_dir, 'for_train', '*', '*.jpg')))

y = nd.zeros((n,))
for i, file_name in tqdm(enumerate(glob(os.path.join('.', data_dir, 'for_train', '*', '*.jpg'))), total=n):
    y[i] = synset.index(file_name.split('/')[3].lower())
    nd.waitall()

#In the commented part below are the features extracted by different combos of pre-trained dnns. Features from differet dnns will be concatenated on top of each other
features = [nd.array(np.load(os.path.join(data_dir, 'features_nas_331.npy'))),nd.array(np.load(os.path.join(data_dir, 'features_incepres.npy')))]#,nd.array(np.load(os.path.join(data_dir, 'features_resnext.npy'))),nd.load(os.path.join(data_dir, 'features_incep_364.nd'))[0]]#,nd.load(os.path.join(data_dir, 'features_res_300.nd'))[0]]#,nd.array(np.load(os.path.join(data_dir, 'features_incepv4.npy')))[0]],nd.load(os.path.join(data_dir, 'features_desnet201.nd'))[0]]#,nd.load(os.path.join(data_dir, 'features_mobilenetv2_1.0.nd'))[0],nd.load(os.path.join(data_dir, 'features_vgg11.nd'))[0],nd.load(os.path.join(data_dir, 'features_AlexNet.nd'))[0]]
features = nd.concat(*features, dim=1)
print(features.shape)

#In the commented part below are the combos of dnn models that have been tested before
models   = ['nas_331','incpres']#,'resnext','incep_364']#,'res_300']#,'incepv4']#,'dense']#,'mobile']#,'vgg']#,'alex']
features_test = [nd.load(os.path.join(data_dir, 'features_test_%s.nd') % model)[0] for model in models]
features_test = nd.concat(*features_test, dim=1)
print(features_test.shape)

#Build the model with the output of 120 classes
def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120,use_bias=True))

    net.initialize(ctx=ctx)
    return net

def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()

#Use Logloss for evaluation of the validation results
def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps

#Shuffle the training set and then divide it into training part and validation part at a ratio of 4:1
#Use the validation to prevent overfitting and find a range for epochs, and then recombine training part and validation part for more training data.
permutation = np.random.permutation(features.shape[0])
x_train = features[permutation][:-int(features.shape[0]//5)]
x_val = features[permutation][-int(features.shape[0]//5):]
y_train = y[permutation][:-int(y.shape[0]//5)]
y_val = y[permutation][-int(y.shape[0]//5):]


data_iter_train       = gluon.data.DataLoader(gluon.data.ArrayDataset(x_train, y_train), batch_size, shuffle=True)
data_iter_val      = gluon.data.DataLoader(gluon.data.ArrayDataset(x_val, y_val), batch_size, shuffle=True)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
net                   = build_model()
trainer               = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

#Mini-batch Gradient Descent
for epoch in range(32):#32
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    val_loss, val_acc = evaluate(net, data_iter_val)

#First uncomment the following two lines and don't put a range for epochs. We will stop the loop once validation loss hits minimum and use that epoch count as the range. Next, comment out the next two lines and rerun mini-batch gradient descent with the whole training set (training + validation) after shuffling it again

#print("Epoch %d., val_loss %.4f, val_acc %.2f%%, train_acc %.2f%%" % (
#           epoch+1, val_loss, val_acc*100, train_acc/steps*100))

output = nd.softmax(net(nd.array(features_test).as_in_context(ctx)))

#The following block is our attempt of implementing pseudo labelling (data augmentation) that will put prediction results that has more than 99% accurary to the training set for re-training. This way, we can increase the size of training data
#Unfortunately, as of the submission of our report, we haven't been able to achieved a satisfied result with pseudo labelling

#t=nd.argmax(output, axis=1)
#t_cpu = (t.copyto(mx.cpu())).asnumpy()
#maxval=nd.max(output,axis=1)
#print(maxval)
#indx=maxval>0.99
#indxx=(indx.copyto(mx.cpu())).asnumpy()
#t_trim=t_cpu[indxx==1]
#temp=features_test.asnumpy()
#f_trim=temp[indxx==1]
#print(t_trim.shape)
#print(f_trim.shape)
#
#labels_all = nd.concat(y,nd.array(t_trim), dim=0)
#feat_all=nd.concat(features,nd.array(f_trim),dim=0)

#Format the prediction results in the given sample_submission.csv file for submission on Kaggle
output=output.asnumpy()
df_pred = pd.read_csv(os.path.join('.', data_dir, 'sample_submission.csv'))

for i, c in enumerate(df_pred.columns[1:]):
        df_pred[c] = output[:,i]
df_pred.to_csv(os.path.join('.', data_dir,str(models[0])+str(epoch)+submit_fileName), index=None)
