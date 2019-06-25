import argparse
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers
import numpy as np
import pandas as pd
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from keras.models import model_from_json
from common_function import read_data_apply, calc_sig, prepare_data
import config_OPT_NN as conf

def calculate_pred(model,X):
    prob_predict=model.predict(X.values, verbose=False)
    pcutNN = np.percentile(prob_predict,40.)
    Yhat=prob_predict[:] > pcutNN
    return Yhat, prob_predict

def save_file(data, pred, proba, filename, model):
    data['isSignal'] = pred
    print(filename)
    data['probSignal'] = proba[:]
    array2root(np.array(data.to_records()), 'OutputRoot/new_'+model+'_'+filename, 'nominal', mode='recreate')
    print('Save file as {}'.format('new_'+model+'_'+filename))
    return

def analyze_data(filedir,filename, model, X_mean, X_dev, label, variables, sigmodel):
    data, X = read_data_apply(filedir+filename, X_mean, X_dev, label, variables, sigmodel)
    pred, proba = calculate_pred(model,X)
    save_file(data, pred, proba, filename, sigmodel)

"""Run Trained Neural Network on samples
Usage:
  python3 Apply_NN.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --input =<input>    Specify input name of trained NN
  --model =<model> Specify signal model ('HVT' or 'GM')
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Apply NN on ntuples')
    parser.add_argument("--input", help="Name of saved trained NN", default='GM_output_NN.h5', type=str)
    parser.add_argument("--model", help="Specify Model (HVT or GM)", default='GM', type=str)

    args = parser.parse_args()
    print(args)

    #Load input_sample class from config file
    input_sample=conf.input_samples
    apply_sample=conf.apply_samples

    #Restores Model and compiles automatically
    model = load_model('OutputModel/'+args.input)
    model.summary()

    #Load Mean and std dev
    if args.model=='GM':
        X_mean = np.load('meanGM.npy')
        X_dev = np.load('std_devGM.npy')
    elif args.model=='HVT':
        X_mean = np.load('meanHVT.npy')
        X_dev = np.load('std_devHVT.npy')
    else :
        raise NameError('Model needs to be either GM or HVT')
    #Mean and std dev from training
    #print(X_mean)
    #print(X_dev)

    #Apply NN on all samples in config file
    list_bkg = apply_sample.list_apply_bkg
    if args.model=='GM': 
        list_sig = apply_sample.list_apply_sigGM
    elif args.model=='HVT':
        list_sig = apply_sample.list_apply_sigHVT  
    print('Applying on bkg sample')
    for i in range(len(list_bkg)):
        analyze_data(apply_sample.filedirbkg,list_bkg[i],model, X_mean, X_dev,-1,input_sample.variables,args.model)
    print('Applying on sig sample')
    for i in range(len(list_sig)):
        analyze_data(apply_sample.filedirsig,list_sig[i],model, X_mean, X_dev,i,input_sample.variables,args.model)
