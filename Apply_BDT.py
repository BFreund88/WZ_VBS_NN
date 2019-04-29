import sklearn
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
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
import config_OPT_BDT as conf

def calculate_pred(model,X):
    prob_predict=model.predict_proba(X)
    pcutNN = 0.0
    Yhat=prob_predict[:,0] > pcutNN
    return Yhat, prob_predict

def save_file(data, pred, proba, filename):
    data['isSignal'] = pred
    print(filename)
    for index in range(20):
        print "Proba {}".format(proba[index,0])
    data['probSignal'] = proba[:,0]
    array2root(np.array(data.to_records()), 'OutputRoot/new_BDT_'+filename, 'nominal', mode='recreate')
    return

def analyze_data(filedir,filename,model, X_mean, X_dev, label, variables):
    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label, variables)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename)

#Load input_sample class from config file
input_sample=conf.input_samples
apply_sample=conf.apply_samples


#Restores Model and compiles automatically
model = joblib.load('./OutputModel/modelBDT_train.pkl')
print(model)



#Load Mean and std dev
X_mean = np.load('mean.npy')
X_dev = np.load('std_dev.npy')

#Apply NN on all samples in config file
list_bkg = apply_sample.list_apply_bkg
list_sig = apply_sample.list_apply_sig
print('Applying on bkg sample')
for i in range(len(list_bkg)):
    analyze_data(apply_sample.filedirbkg,list_bkg[i],model, X_mean, X_dev,-1,input_sample.variables)
print('Applying on sig sample')
for i in range(len(list_sig)):
    analyze_data(apply_sample.filedirsig,list_sig[i],model, X_mean, X_dev,i+1,input_sample.variables)
