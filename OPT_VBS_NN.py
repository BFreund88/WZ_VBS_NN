import argparse
import sys
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
import matplotlib.pyplot as plt
import math
from root_numpy import root2array, tree2array, array2root
import ROOT
from common_function import dataset, AMS, read_data, prepare_data, drawfigure
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
  --numlayer=<numlayer>     Specify number of hidden layers.
  --numn=<numn> Number of neurons per hidden layer
  --dropout=<dropout> Dropout to reduce overfitting
  --epoch=<epochs> Specify training epochs
  --lr=<lr> Learning rate for SGD optimizer
"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'NN optimisation')
    parser.add_argument("-v", "--verbose", action="store_true", help="increase output verbosity")
    parser.add_argument('--numlayer', help = "Specifies the number of layers of the Neural Network", default=3, type=int)
    parser.add_argument('--numn', help = "Specifies the number of neurons per hidden layer", default=200, type=int)
    parser.add_argument('--lr','--learning_rate', help = "Specifies the learning rate for SGD optimizer", default=0.01, type=float)
    parser.add_argument('--dropout', help = "Specifies the applied dropout", default=0.05, type=float)
    parser.add_argument('--epochs', help = "Specifies the number of epochs", default=80, type=int)
    
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

    #Define checkpoint to save best performing NN
    checkpoint=keras.callbacks.ModelCheckpoint(filepath='output_NN.h5', monitor='val_acc', verbose=args.verbose, save_best_only=True)
    
    #Train Model
    logs = model.fit(data_set.X_train, data_set.y_train, epochs=args.epochs,
                     validation_data=(data_set.X_valid, data_set.y_valid),batch_size=128, callbacks=[checkpoint], verbose =1, class_weight = 'auto')

    plt.plot(logs.history['acc'], label='train')
    plt.plot(logs.history['val_acc'], label='valid')
    plt.legend()
    plt.title('Pourcentage of correct classification')
    plt.savefig('class.png')
    plt.clf()
    plt.plot(logs.history['loss'], label='train')
    plt.plot(logs.history['val_loss'], label='valid')
    plt.legend()
    plt.title('Training loss')
    plt.savefig('loss.png')
    plt.clf() 

    #Calculate Significance
    #Load saved weights which gave best result on training
    model.load_model('output_NN.h5')

    prob_predict_train_NN = model.predict(data_set.X_train, verbose=False)
    prob_predict_valid_NN = model.predict(data_set.X_valid, verbose=False)
    #prob_predict_test_NN = model.predict(data_set.X_test, verbose=False)

    #lower=470
    lower=470
    upper=540
    massindex=0
    mass=300

    drawfigure(model,prob_predict_train_NN,data_set,data_set.X_valid)

    AMS_train=np.zeros((upper-lower,2))
    AMS_valid=np.zeros((upper-lower,2))
    index2=0

    #for index in range(200):
    #    print "Label {}".format(data_set.y_train[index,1])
    #    print "Proba {}".format(prob_predict_train_NN[index,:])


    for loop2 in range(lower,upper):

        print "With upper percentile {}".format(loop2/10.)
        pcutNN = np.percentile(prob_predict_train_NN,loop2/10.)

        print("Cut Value {}".format(pcutNN))
        Yhat_train_NN = prob_predict_train_NN[:,1] > pcutNN
        Yhat_valid_NN = prob_predict_valid_NN[:,1] > pcutNN
    
        s_train_NN=b_train_NN=0
        s_valid_NN=b_valid_NN=0

        for index in range(len(Yhat_train_NN)):
            if (Yhat_train_NN[index]==1.0 and data_set.y_train[index,1]==1 and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train_label.iloc[index,0]<mass+mass*0.08*1.5 and data_set.mass_train_label.iloc[index,0]==massindex):
                s_train_NN +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))
            elif (Yhat_train_NN[index]==1.0 and data_set.y_train[index,1]==0 and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5):
                b_train_NN +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))

        for index in range(len(Yhat_valid_NN)):
            if (Yhat_valid_NN[index]==1.0 and data_set.y_valid[index,1]==1 and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid_label.iloc[index,0]<mass+mass*0.08*1.5 and data_set.mass_valid_label.iloc[index,0]==massindex):
                s_valid_NN +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))
            elif (Yhat_valid_NN[index]==1.0 and data_set.y_valid[index,1]==0 and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid.iloc[index,0]<mass+mass*0.08*1.5):
                b_valid_NN +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))

        print "S and B NN training"
        print s_train_NN
        print b_train_NN
        print "S and B NN validation"
        print s_valid_NN
        print b_valid_NN

        print 'Calculating AMS score for NNs with a probability cutoff pcut=',pcutNN
        print '   - AMS based on 90% training   sample:',AMS(s_train_NN,b_train_NN)
        print '   - AMS based on 10% validation sample:',AMS(s_valid_NN,b_valid_NN)
    
        AMS_train[index2,0]=loop2/10.
        AMS_train[index2,1]=AMS(s_train_NN,b_train_NN)
        AMS_valid[index2,0]=loop2/10.
        AMS_valid[index2,1]=AMS(s_valid_NN,b_valid_NN)
        index2=index2+1

        plt.plot(AMS_train[:,0],AMS_train[:,1], label='train')
        plt.plot(AMS_valid[:,0],AMS_valid[:,1], label='valid')
        plt.legend()
        plt.title('Significance as a function of the probability output')
        plt.savefig('significance_NN.png')

