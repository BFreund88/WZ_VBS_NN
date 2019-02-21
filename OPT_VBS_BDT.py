from keras.utils.np_utils import to_categorical
import sklearn
from sklearn.externals import joblib
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from root_numpy import root2array, tree2array, array2root
import ROOT
import matplotlib.pyplot as plt
from common_function import dataset, AMS, read_data, prepare_data

def BDTModelada(max_depth, learning_rate, n_estimators, algorithm):
    BDTada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                         learning_rate=learning_rate,
                         n_estimators=n_estimators,
                         algorithm=algorithm)
    return BDTada

def BDTModelgrad(max_depth, learning_rate, n_estimators,verbose,n_iter_no_change):
    BDTgrad = GradientBoostingClassifier(verbose=verbose,\
                                             n_iter_no_change=n_iter_no_change,
                                         learning_rate=learning_rate,
                                         n_estimators=n_estimators
        )
    return BDTgrad


#Directory where the ntuples are located
filedir = '/lcg/storage15/atlas/freund/ntuples_Miaoran/'
print('Read files from directory {0}'.format(filedir))
#Assumed luminosity
lumi=150.

 
data_set=prepare_data(filedir, lumi)

max_depth = 3
learning_rate = 0.2
n_estimators = 800
algorithm = "SAMME.R"
verbose = 1
n_iter_no_change = 20

#model=BDTModelada(max_depth, learning_rate, n_estimators, algorithm)
model=BDTModelgrad(max_depth, learning_rate, n_estimators,verbose,n_iter_no_change)


shape_train=data_set.X_train.shape
shape_valid=data_set.X_valid.shape
#shape_test=data_set.X_test.shape

num_train=shape_train[0]
num_valid=shape_valid[0]
#num_test=shape_test[0]
num_tot=num_train+num_valid#+num_test


print("The number of training events {0} validation events {1} and total events {2}".format(num_train,num_valid,num_tot))


print(model)

print("Fit Model")

model.fit(data_set.X_train, data_set.y_train[:,0].ravel())


print("Save Model")

filenameBDT = './modelBDT_train.pkl'
_ = joblib.dump(model, filenameBDT, compress=9)

# Plot the two-class decision scores
plot_colors = "br"
plot_step = 0.025
twoclass_output = model.decision_function(data_set.X_train)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(111)
class_names = "SB"

print('Save decision plot')

for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(twoclass_output[data_set.y_train[:,0] == i],
             bins=20,
             range=plot_range,
             facecolor=c,
             label='Class %s' % n,
             alpha=.5,
             #edgecolor='k',
             log=True)
x1, x2, y1, y2 = plt.axis()

# Make a colorful backdrop to show the clasification regions in red and blue
plt.axvspan(0, twoclass_output.max(), color='blue',alpha=0.08)
plt.axvspan(twoclass_output.min(),0, color='red',alpha=0.08)

plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc='upper right')
plt.ylabel('Samples')
plt.xlabel('Score')
plt.title('Decision Scores')

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.savefig('./ControlPlots/Decision_score_BDT.png')

lower=-0.2
upper=0.1
stepsize=0.02

AMS_train=np.zeros((int((upper-lower)/stepsize),2))
AMS_valid=np.zeros((int((upper-lower)/stepsize),2))

index2=0

prob_predict_train_BDT = model.decision_function(data_set.X_train)
prob_predict_valid_BDT = model.decision_function(data_set.X_valid)

mass=300

for loop2 in range(0,int((upper-lower)/stepsize)):

    print "Cutting value {}".format(loop2*stepsize+lower)
    
    Yhat_train_BDT = prob_predict_train_BDT[:] > loop2*stepsize+lower
    Yhat_valid_BDT = prob_predict_valid_BDT[:] > loop2*stepsize+lower
    
    s_train_BDT=b_train_BDT=0
    s_valid_BDT=b_valid_BDT=0

    for index in range(len(Yhat_train_BDT)):
        if (Yhat_train_BDT[index]==1.0 and data_set.y_train[index,1]==1 and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5):
            s_train_BDT +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))
        elif (Yhat_train_BDT[index]==1.0 and data_set.y_train[index,1]==0 and data_set.mass_train.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_train.iloc[index,0]<mass+mass*0.08*1.5):
            b_train_BDT +=  abs(data_set.W_train.iat[index,0]*(num_tot/float(num_train)))

    for index in range(len(Yhat_valid_BDT)):
        if (Yhat_valid_BDT[index]==1.0 and data_set.y_valid[index,1]==1 and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid.iloc[index,0]<mass+mass*0.08*1.5):
            s_valid_BDT +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))
        elif (Yhat_valid_BDT[index]==1.0 and data_set.y_valid[index,1]==0 and data_set.mass_valid.iloc[index,0]>mass-mass*0.08*1.5 and data_set.mass_valid.iloc[index,0]<mass+mass*0.08*1.5):
            b_valid_BDT +=  abs(data_set.W_valid.iat[index,0]*(num_tot/float(num_valid)))

    print "S and B NN training"
    print s_train_BDT
    print b_train_BDT
    print "S and B NN validation"
    print s_valid_BDT
    print b_valid_BDT

    print 'Calculating AMS score for BDTs with a probability cutoff pcut=',loop2*stepsize+lower
    print '   - AMS based on 90% training   sample:',AMS(s_train_BDT,b_train_BDT)
    print '   - AMS based on 10% validation sample:',AMS(s_valid_BDT,b_valid_BDT)
    
    AMS_train[index2,0]=loop2*stepsize+lower
    AMS_train[index2,1]=AMS(s_train_BDT,b_train_BDT)
    AMS_valid[index2,0]=loop2*stepsize+lower
    AMS_valid[index2,1]=AMS(s_valid_BDT,b_valid_BDT)
    index2=index2+1

plt.plot(AMS_train[:,0],AMS_train[:,1], label='train')
plt.plot(AMS_valid[:,0],AMS_valid[:,1], label='valid')
plt.legend()
plt.title('Significance as a function of the probability output')
plt.savefig('./ControlPlots/significance_BDT.png')

