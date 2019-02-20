import sklearn
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib.pyplot as plt
from keras.models import model_from_json
from common_function import read_data_apply

def calculate_pred(model,X):
    prob_predict=model.predict_proba(X)
    pcutNN = 0.5
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

def analyze_data(filedir,filename,model, X_mean, X_dev, label):
    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename)

#Restores Model and compiles automatically
model = joblib.load('./modelBDT_train.pkl')
print(model)

filedir = '/lcg/storage15/atlas/freund/ntuples_Miaoran/'
filename1= '364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root'
filename2= '364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root'

filedirsig=filedir+'new/signalsNew/mc16a/'
filenamsig300= '305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_Systematics.root'
filenamsig400= '305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_Systematics.root'
filenamsig500= '305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_Systematics.root'
filenamsig600= '305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_Systematics.root'
filenamsig700= '305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_Systematics.root'
filenamsig800= '305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_Systematics.root'
filenamsig900= '305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_Systematics.root'

#Load Mean and std dev
X_mean = np.load('mean.npy')
X_dev = np.load('std_dev.npy')

print(X_mean)
print(X_dev)

#Read files
analyze_data(filedir,filename1,model, X_mean, X_dev,-1)
analyze_data(filedir,filename2,model, X_mean, X_dev,-1)
analyze_data(filedirsig,filenamsig300,model, X_mean, X_dev,0)
analyze_data(filedirsig,filenamsig400,model, X_mean, X_dev,1)
analyze_data(filedirsig,filenamsig500,model, X_mean, X_dev,2)
analyze_data(filedirsig,filenamsig600,model, X_mean, X_dev,3)
analyze_data(filedirsig,filenamsig700,model, X_mean, X_dev,4)
analyze_data(filedirsig,filenamsig800,model, X_mean, X_dev,5)
analyze_data(filedirsig,filenamsig900,model, X_mean, X_dev,6)
