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
import sklearn
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
from common_function import dataset, AMS, read_data, prepare_data, drawfigure, calc_sig
import config_OPT_NN as conf

def KerasModel(input_dim,numlayer,numn, dropout):
    model = Sequential()
    model.add(Dense(numn, input_dim=int(input_dim)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    for i in range(numlayer-1):
        model.add(Dense(numn))
        model.add(Activation('relu'))
        model.add(Dropout(dropout))
    model.add(Dense(2))
    model.add(Activation('sigmoid'))

    return model

"""Neural Network Optimisation

Usage:
  OPT_VBS_NN.py 

Options:
  -h --help             Show this screen.
Optional arguments
  --v = <verbose> Set verbose level
  --output =<output>    Specify output name
  --numlayer=<numlayer>     Specify number of hidden layers.
  --numn=<numn> Number of neurons per hidden layer
  --dropout=<dropout> Dropout to reduce overfitting
  --epoch=<epochs> Specify training epochs
  --patience=<patience> Set patience for early stopping
  --lr=<lr> Learning rate for SGD optimizer
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'NN optimisation')
    parser.add_argument("--v", "--verbose", help="increase output verbosity", default=0, type=int)
    parser.add_argument("--output", help="Specify Output name", default='', type=str)
    parser.add_argument('--numlayer', help = "Specifies the number of layers of the Neural Network", default=3, type=int)
    parser.add_argument('--numn', help = "Specifies the number of neurons per hidden layer", default=200, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate for SGD optimizer", default=0.01, type=float)
    parser.add_argument('--dropout', help = "Specifies the applied dropout", default=0.05, type=float)
    parser.add_argument('--epochs', help = "Specifies the number of epochs", default=80, type=int)
    parser.add_argument('--patience', help = "Specifies the patience for early stopping", default=5, type=int)

    
    args = parser.parse_args()
    print(args)

    #Load input_sample class from config file
    input_sample=conf.input_samples

    #Read data files
    data_set=prepare_data(input_sample)
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
    model=KerasModel(shape_train[1],args.numlayer,args.numn,args.dropout)
    
    #Possible optimizers
    sgd = optimizers.SGD(lr=args.lr, decay=1e-6, momentum=0.6, nesterov=True)
    ada= optimizers.Adadelta(lr=1, rho=0.95, epsilon=None, decay=0.01)
    nadam=keras.optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.save("modelNN_initial.h5")

    # Save the model architecture
    with open('model_architecture.json', 'w') as f:
        f.write(model.to_json())

    #Define checkpoint to save best performing NN and early stopping
    path='./OutputModel/'+args.output+'output_NN.h5'
    #checkpoint=keras.callbacks.ModelCheckpoint(filepath='output_NN.h5', monitor='val_acc', verbose=args.v, save_best_only=True)
    callbacks=[EarlyStopping(monitor='val_loss', patience=args.patience),ModelCheckpoint(filepath=path, monitor='val_loss', verbose=args.v, save_best_only=True)]
    
    #Train Model
    logs = model.fit(data_set.X_train, data_set.y_train, epochs=args.epochs,
                     validation_data=(data_set.X_valid, data_set.y_valid),batch_size=256, callbacks=callbacks, verbose =args.v, class_weight = 'auto')

    plt.plot(logs.history['acc'], label='train')
    plt.plot(logs.history['val_acc'], label='valid')
    plt.legend()
    plt.title('')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig('./ControlPlots/class.png')
    plt.clf()
    plt.plot(logs.history['loss'], label='train')
    plt.plot(logs.history['val_loss'], label='valid')
    plt.legend()
    #plt.title('Training loss')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title('')
    
    plt.savefig('./ControlPlots/loss.png')
    plt.clf() 

    #Calculate Significance
    #Load saved weights which gave best result on training
    model = load_model('output_NN.h5')

    prob_predict_train_NN = model.predict(data_set.X_train, verbose=False)
    prob_predict_valid_NN = model.predict(data_set.X_valid, verbose=False)
    #prob_predict_test_NN = model.predict(data_set.X_test, verbose=False)

    #Draw same figures
    drawfigure(model,prob_predict_train_NN,data_set,data_set.X_valid)

    #for index in range(200):
    #    print "Label {}".format(data_set.y_train[index,1])
    #    print "Proba {}".format(prob_predict_train_NN[index,:])

    #Calculate significance in output range between lower and upper
    lower=40
    upper=70
    massindex=0
    mass=300
    step = 2
    
    calc_sig(data_set, prob_predict_train_NN[:,1], prob_predict_valid_NN[:,1], lower,upper,step,mass,massindex,'NN')

