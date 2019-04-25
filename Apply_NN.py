import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers
import sklearn
import numpy as np
import pandas as pd
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
from common_function import read_data_apply
import config_OPT_NN as conf

def calculate_pred(model,X):
    prob_predict=model.predict(X, verbose=False)
    pcutNN = np.percentile(prob_predict,40.)
    Yhat=prob_predict[:,1] > pcutNN
    return Yhat, prob_predict

def save_file(data, pred, proba, filename):
    data['isSignal'] = pred
    print(filename)
    #for index in range(20):
    #    print "Proba {}".format(proba[index,:])
    data['probSignal'] = proba[:,1]
    array2root(np.array(data.to_records()), 'OutputRoot/new_'+filename, 'nominal', mode='recreate')
    return

def analyze_data(filedir,filename,model, X_mean, X_dev, label, variables):
    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label, variables)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename)

#Load input_sample class from config file
input_sample=conf.input_samples
apply_sample=conf.apply_samples

#Restores Model and compiles automatically
model = load_model('output_NN.h5')
model.summary()

#Load Mean and std dev
X_mean = np.load('mean.npy')
X_dev = np.load('std_dev.npy')

#Mean and std dev from training
#print(X_mean)
#print(X_dev)

#Apply NN on all samples in config file
list_bkg = apply_sample.list_apply_bkg
list_sig = apply_sample.list_apply_sig
print('Applying on bkg sample')
for i in range(len(list_bkg)):
    analyze_data(apply_sample.filedirbkg,list_bkg[i],model, X_mean, X_dev,-1,input_sample.variables)
print('Applying on sig sample')
for i in range(len(list_sig)):
    analyze_data(apply_sample.filedirsig,list_sig[i],model, X_mean, X_dev,i+1,input_sample.variables)

