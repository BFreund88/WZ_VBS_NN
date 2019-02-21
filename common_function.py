import keras
from keras.utils.np_utils import to_categorical
import ROOT
from root_numpy import root2array, tree2array, array2root
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

def AMS(s, b):
    """ Approximate Median Significance defined as:
        AMS = sqrt(
                2 { (s + b + b_r) log[1 + (s/(b+b_r))] - s}
              )        
    where b_r = 0.00001, b = background, s = signal, log is natural logarithm with added systematics"""
    
    br = 0.00001
    sigma=math.sqrt(b+br)
    n=s+b+br
    radicand = 2 *( n * math.log (n*(b+sigma)/(b**2+n*sigma+br))-b**2/sigma*math.log(1+sigma*(n-b)/(b*(b+br+sigma))))
    if radicand < 0:
        print 'radicand is negative. Exiting'
        exit()
    else:
        return math.sqrt(radicand)

def read_data_apply(filepath, X_mean, X_dev, Label, variables):
    data = read_data(filepath)
    data = data.reset_index(drop=True)
    data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    X = data[variables]

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


def read_data(filename):
    root = ROOT.TFile(filename)
    tree = root.Get('nominal')
    array = tree2array(tree, selection='m_Valid_jet1==1 && m_Valid_jet2==1')
    return pd.DataFrame(array)

class dataset:
    def __init__(self,data,frac_train,frac_valid,variables):
        train_full=data.sample(frac=frac_train,random_state=42)
        #test=data.drop(train_full.index)
        train=train_full.sample(frac=frac_valid,random_state=42)
        validation=train_full.drop(train.index)

        #Separate variables from labels
        self.y_train=to_categorical(train[['Label']])
        self.y_valid=to_categorical(validation[['Label']])
        #self.y_test=to_categorical(test[['Label']])

        mass_train=train[['Mass']]
        mass_valid=validation[['Mass']]
        #mass_test=test[['Mass']]

        self.mass_train=mass_train.reset_index(drop=True)
        self.mass_valid=mass_valid.reset_index(drop=True)
        #self.mass_test=mass_test.reset_index(drop=True)

        self.W_train=train[['Weight']]
        self.W_valid=validation[['Weight']]
        #self.W_test=test[['Weight']]

        X_train = train[variables]
        X_valid = validation[variables]

        #Save mean and std dev
        np.save('./mean', np.mean(X_train))
        np.save('./std_dev', np.std(X_train))

        self.X_train= X_train-np.mean(X_train)
        self.X_train= X_train/np.std(X_train)

        self.X_valid= X_valid-np.mean(X_valid)
        self.X_valid= X_valid/np.std(X_valid)
        
        #self.X_test= self.X_test-np.mean(self.X_test)
        #self.X_test= self.X_test/np.std(self.X_test)

        self.X_train['LabelMass']=train[['LabelMass']]
        self.X_valid['LabelMass']=validation[['LabelMass']]

        self.mass_train_label=train[['LabelMass']]
        self.mass_valid_label=validation[['LabelMass']]
        #self.X_test['LabelMass']=test[['LabelMass']]
        
def prepare_data(input_samples):
    #Read background and signal files and save them as panda data frames

    #Names of bck samples
    namesbkg = input_samples.bckgr["name"]
    xsbkg = input_samples.bckgr["xs"]
    neventsbkg = input_samples.bckgr["nevents"]
    #Read files one by one and normalize weights to 150 fb^-1
    bg = None
    print('Read Background Samples')
    for i in range(len(namesbkg)):
        sample = read_data(input_samples.filedir+namesbkg[i])
        print(namesbkg[i])
        sample['Weight']=sample['Weight']*input_samples.lumi*xsbkg[i]/neventsbkg[i]
        if bg is None:
            bg=sample
        else:
            bg=bg.append(sample, sort=True)

    #Add label 0 for bkg
    bg['Label'] = '0'

    #Read signal
    namessig = input_samples.sig["name"]
    xssig = input_samples.sig["xs"]
    neventssig = input_samples.sig["nevents"]
    sig = None
    prob = np.empty(len(namessig))
    print('Read Signal Samples')
    for i in range(len(namessig)):
        sample = read_data(input_samples.filedirsig+namessig[i])
        print(namessig[i])
        sample['Weight']=sample['Weight']*input_samples.lumi*xssig[i]/neventssig[i]
        sample['LabelMass'] = i+1
        prob[i] = sample.shape[0] 
        if sig is None:
            sig=sample
        else:
            sig=sig.append(sample, sort=True)
    #Probability distribution for random Mass Label
    prob=prob/float(sig.shape[0])
    sig['Label'] = '1'

    #Apply random mass label to bkg
    label=np.random.choice(7,bg.shape[0], p=prob)

    bg['LabelMass'] = label

    #Save prob distribution
    np.save('./prob', prob)
    
    data=bg.append(sig, sort=True)
    data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    data = data.sample(frac=1).reset_index(drop=True)
    # Pick a random seed for reproducible results
    # Use 30% of the training sample for validation

    data_cont = dataset(data,1.,input_samples.valfrac,input_samples.variables)
    return data_cont

def drawfigure(model,prob_predict_train_NN,data,X_test):
    pcutNN = np.percentile(prob_predict_train_NN,500/10.)

    Classifier_training_S = model.predict(data.X_train[data.y_train[:,1]==1], verbose=False)[:,1].ravel()
    Classifier_training_B = model.predict(data.X_train[data.y_train[:,1]==0], verbose=False)[:,1].ravel()
    Classifier_testing_A = model.predict(X_test, verbose=False)[:,1].ravel()

    c_max = max([Classifier_training_S.max(),Classifier_training_B.max(),Classifier_testing_A.max()])
    c_min = min([Classifier_training_S.min(),Classifier_training_B.min(),Classifier_testing_A.min()])
  
    # Get histograms of the classifiers NN
    Histo_training_S = np.histogram(Classifier_training_S,bins=50,range=(c_min,c_max))
    Histo_training_B = np.histogram(Classifier_training_B,bins=50,range=(c_min,c_max))
    Histo_testing_A = np.histogram(Classifier_testing_A,bins=50,range=(c_min,c_max))

    # Lets get the min/max of the Histograms
    AllHistos= [Histo_training_S,Histo_training_B]
    h_max = max([histo[0].max() for histo in AllHistos])*1.2

    # h_min = max([histo[0].min() for histo in AllHistos])
    h_min = 1.0
  
    # Get the histogram properties (binning, widths, centers)
    bin_edges = Histo_training_S[1]
    bin_centers = ( bin_edges[:-1] + bin_edges[1:]  ) /2.
    bin_widths = (bin_edges[1:] - bin_edges[:-1])

    # To make error bar plots for the data, take the Poisson uncertainty sqrt(N)
    ErrorBar_testing_A = np.sqrt(Histo_testing_A[0])

    # Draw objects
    ax1 = plt.subplot(111)
  
    # Draw solid histograms for the training data
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_B[0],facecolor='red',linewidth=0,
            width=bin_widths,label='B (Train)',alpha=0.5)
    ax1.bar(bin_centers-bin_widths/2.,Histo_training_S[0],bottom=Histo_training_B[0],
            facecolor='blue',linewidth=0,width=bin_widths,label='S (Train)',alpha=0.5)
 
    ff = 1.5*(1.0*(sum(Histo_training_S[0])+sum(Histo_training_B[0])))/(1.0*sum(Histo_testing_A[0]))
 
     # # Draw error-bar histograms for the testing data
    ax1.errorbar(bin_centers-bin_widths/2, ff*Histo_testing_A[0], yerr=ff*ErrorBar_testing_A, xerr=None, 
                 ecolor='black',c='black',fmt='.',label='Test (reweighted)')
  
    # Make a colorful backdrop to show the clasification regions in red and blue
    ax1.axvspan(pcutNN, c_max, color='blue',alpha=0.08)
    ax1.axvspan(c_min,pcutNN, color='red',alpha=0.08)
  
    # Adjust the axis boundaries (just cosmetic)
    ax1.axis([c_min, c_max, h_min, h_max])
  
    # Make labels and title
    plt.title("")
    plt.xlabel("Probability Output (NN)")
    plt.ylabel("Counts/Bin")
    plt.yscale('log', nonposy='clip')
 
    # Make legend with smalll font
    legend = ax1.legend(loc='upper center', shadow=True,ncol=2)
    for alabel in legend.get_texts():
        alabel.set_fontsize('small')
  
    # Save the result to png
    plt.savefig("./ControlPlots/NN_clf.png")
    plt.clf() 


def calc_sig(lower,upper,step,mass,massindex):
    AMS_train=np.zeros((upper-lower,2))
    AMS_valid=np.zeros((upper-lower,2))
    index2=0

    for loop2 in range(lower,upper, step):
        print "With upper percentile {}".format(loop2)
        pcutNN = np.percentile(prob_predict_train_NN,loop2)

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
    
        AMS_train[index2,0]=loop2
        AMS_train[index2,1]=AMS(s_train_NN,b_train_NN)
        AMS_valid[index2,0]=loop2
        AMS_valid[index2,1]=AMS(s_valid_NN,b_valid_NN)
        index2=index2+1

    plt.plot(AMS_train[:,0],AMS_train[:,1], label='train')
    plt.plot(AMS_valid[:,0],AMS_valid[:,1], label='valid')
    plt.legend()
    plt.title('Significance as a function of the probability output')
    plt.savefig('./ControlPlots/significance_NN.png')
