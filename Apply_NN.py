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
import matplotlib.pyplot as plt
from keras.models import model_from_json

def read_data(filepath, X_mean, X_dev, Label):
    rootfile=ROOT.TFile(filepath)
    tree = rootfile.Get('nominal')
    array = tree2array(tree, selection='m_Valid_jet1==1 && m_Valid_jet2==1')
    data = pd.DataFrame(array)
    data = data.reset_index(drop=True)
    data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    X = data[['Mjj','Detajj','MET','PtZoverWZmass','PtWoverWZmass' ,'m_Pt_jet1','m_Pt_jet2', \
                     'm_Pt_jet3','m_Eta_jet1','m_Eta_jet2','m_Eta_jet3','m_E_jet1','m_E_jet2','m_E_jet3',\
                     'm_Eta_lep1','m_Eta_lep2', 'm_Eta_lep3','m_Pt_lep1', 'm_Pt_lep2','m_Pt_lep3','m_Pt_W',\
                     'm_Pt_Z']]

    X= X-X_mean
    X= X/X_dev
    if (Label>-1):
        X['LabelMass']=Label
    else:
        prob=np.load('prob.npy')
        label=np.random.choice(7,X.shape[0], p=prob)
        X['LabelMass'] = label

    data['LabelMass']=X['LabelMass'] 
    return data, X

def calculate_pred(model,X):
    prob_predict=model.predict(X, verbose=False)
    pcutNN = np.percentile(prob_predict,40.)
    Yhat=prob_predict[:,1] > pcutNN
    return Yhat, prob_predict

def save_file(data, pred, proba, filename):
    data['isSignal'] = pred
    print(filename)
    for index in range(100):
        print "Proba {}".format(proba[index,:])
    data['probSignal'] = proba[:,1]
    array2root(np.array(data.to_records()), 'OutputRoot/new_'+filename, 'nominal', mode='recreate')
    return

def analyze_data(filedir,filename,model, X_mean, X_dev, label):
    data, X = read_data(filedir+filename, X_mean, X_dev, label)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename)

#Restores Model and compiles automatically
model = load_model('output_NN.h5')
model.summary()

# Load weights into the new model
#model.load_weights('output_NN.h5')

#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=False)
#ada= optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.0)
#model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])

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
