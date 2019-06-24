import argparse
import sys
import keras
from keras.utils.np_utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, Dropout
from keras import backend as K
from keras.optimizers import SGD
from keras import optimizers
from keras.callbacks import EarlyStopping, ModelCheckpoint
#import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from root_numpy import root2array, tree2array, array2root
from common_function import dataset, AMS, read_data, prepare_data, drawfigure, calc_sig
import config_OPT_NN as conf
import ROOT 

def KerasModel(input_dim,numlayer,numn, bool_drop, dropout):
    model = Sequential()
    model.add(Dense(numn, input_dim=int(input_dim)))
    model.add(Activation('relu'))
    if bool_drop: model.add(Dropout(dropout))
    for i in range(numlayer-1):
        model.add(Dense(numn))
        model.add(Activation('relu'))
        if bool_drop: model.add(Dropout(dropout))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    return model

#Atlernative metric to accuracy
def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

"""Neural Network Optimisation
Usage:
  python3 OPT_VBS_NN.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --v = <verbose> Set verbose level
  --model =<model> Specify signal model ('HVT' or 'GM')
  --output =<output>    Specify output name
  --numlayer=<numlayer>     Specify number of hidden layers.
  --numn=<numn> Number of neurons per hidden layer
  --booldropout=<booldropout> Apply dropout or not 
  --dropout=<dropout> Dropout to reduce overfitting
  --epoch=<epochs> Specify training epochs
  --patience=<patience> Set patience for early stopping
  --lr=<lr> Learning rate for SGD optimizer
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'NN optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--model", help="Specify Model (HVT or GM)", default='GM', type=str)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--numlayer', help = "Specifies the number of layers of the Neural Network", default=3, type=int)
    parser.add_argument('--numn', help = "Specifies the number of neurons per hidden layer", default=200, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate for SGD optimizer", default=0.01, type=float)
    parser.add_argument('--booldropout', help = "Applies Dropout or not", default=0, type=bool)
    parser.add_argument('--dropout', help = "Specifies the applied dropout", default=0.05, type=float)
    parser.add_argument('--epochs', help = "Specifies the number of epochs", default=80, type=int)
    parser.add_argument('--patience', help = "Specifies the patience for early stopping", default=5, type=int)

    
    args = parser.parse_args()
    print(args)

    #Load input_sample class from config file
    input_sample=conf.input_samples

    #Additional name from model and hyper parameters
    nameadd=args.output+args.model

    #Read data files
    data_set=prepare_data(input_sample,args.model)
    #Get input dimensions
    shape_train=data_set.X_train.shape
    shape_valid=data_set.X_valid.shape
    #shape_test=data_set.X_test.shape

    num_train=shape_train[0]
    num_valid=shape_valid[0]
    #num_test=shape_test[0]

    num_tot=num_train+num_valid#+num_test
    
    print('Number of training: {}, validation: {} and total events: {}.'.format(num_train,num_valid,num_tot))

    #Define model with given parameters
    model=KerasModel(shape_train[1],args.numlayer,args.numn,args.booldropout,args.dropout)
    
    #Possible optimizers
    sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.6, nesterov=True)
    ada= optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.01)
    nadam=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])#, f1_score])
    model.summary()
    #model.save("modelNN_initial_"+args.model+".h5")

    # Save the model architecture
    #with open('OutputModel/model_architecture.json', 'w') as f:
    #    f.write(model.to_json())

    #Define checkpoint to save best performing NN and early stopping
    path='./OutputModel/'+nameadd+'output_NN.h5'
    #checkpoint=keras.callbacks.ModelCheckpoint(filepath='output_NN.h5', monitor='val_acc', verbose=args.v, save_best_only=True)
    callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience),ModelCheckpoint(filepath=path, monitor='val_loss', verbose=args.v, save_best_only=True)]
    
    #Train Model
    logs = model.fit(data_set.X_train.values, data_set.y_train.values, epochs=args.epochs,
                     validation_data=(data_set.X_valid.values, data_set.y_valid.values),batch_size=256, callbacks=callbacks, verbose =args.v, class_weight = 'auto')

    plt.plot(logs.history['acc'], label='train')
    plt.plot(logs.history['val_acc'], label='valid')
    plt.legend()
    plt.title('')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('./ControlPlots/class_'+nameadd+'.png')
    plt.clf()
    plt.plot(logs.history['loss'], label='train')
    plt.plot(logs.history['val_loss'], label='valid')
    plt.legend()
    #plt.title('Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('')
    
    plt.savefig('./ControlPlots/loss_'+nameadd+'.png')
    plt.clf() 

    #Calculate Significance
    #Load saved weights which gave best result on training
    model = load_model(path)

    prob_predict_train_NN = model.predict(data_set.X_train.values, verbose=False)
    prob_predict_valid_NN = model.predict(data_set.X_valid.values, verbose=False)
    #prob_predict_test_NN = model.predict(data_set.X_test, verbose=False)

    #Draw same figures
    drawfigure(model,prob_predict_train_NN,data_set,data_set.X_valid.values,nameadd)

    #for index in range(200):
    #    print "Label {}".format(data_set.y_train[index,1])
    #    print "Proba {}".format(prob_predict_train_NN[index,:])

    #Calculate significance in output range between lower and upper
    lower=50
    upper=80
    massindex=0
    mass=200
    step = 2
    
    highsig = calc_sig(data_set, prob_predict_train_NN[:,0], prob_predict_valid_NN[:,0], lower, upper, step, mass, massindex,'NN', args.output, args.model)

    #Save model in OutputModel with the highest significance obtained on validation set
    model.save('./OutputModel/sigvalid_'+args.model+'_'+str(round(highsig,3))+'_'+args.output+'best_NN.h5')
