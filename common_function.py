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

def read_data(filename):
    root = ROOT.TFile(filename)
    tree = root.Get('nominal')
    array = tree2array(tree, selection='m_Valid_jet1==1 && m_Valid_jet2==1')
    return pd.DataFrame(array)

class dataset:
    def __init__(self,data,frac_train,frac_valid):
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

        X_train = train[['Mjj','Detajj','MET','PtZoverWZmass','PtWoverWZmass' ,'m_Pt_jet1','m_Pt_jet2', \
                     'm_Pt_jet3','m_Eta_jet1','m_Eta_jet2','m_Eta_jet3','m_E_jet1','m_E_jet2','m_E_jet3',\
                     'm_Eta_lep1','m_Eta_lep2', 'm_Eta_lep3','m_Pt_lep1', 'm_Pt_lep2','m_Pt_lep3','m_Pt_W',\
                     'm_Pt_Z']]

        X_valid = validation[['Mjj','Detajj','MET','PtZoverWZmass','PtWoverWZmass' ,'m_Pt_jet1','m_Pt_jet2', \
                          'm_Pt_jet3','m_Eta_jet1','m_Eta_jet2','m_Eta_jet3','m_E_jet1','m_E_jet2',\
                          'm_E_jet3', 'm_Eta_lep1','m_Eta_lep2',  \
                          'm_Eta_lep3','m_Pt_lep1', 'm_Pt_lep2','m_Pt_lep3','m_Pt_W', 'm_Pt_Z']]

        #self.X_test = test[['Mjj','Detajj','MET','PtZoverWZmass','PtWoverWZmass' ,'m_Pt_jet1','m_Pt_jet2', \
#                   'm_Pt_jet3','m_Eta_jet1','m_Eta_jet2','m_Eta_jet3','m_E_jet1','m_E_jet2','m_E_jet3',\
#                   'm_Eta_lep1','m_Eta_lep2', \
#                   'm_Eta_lep3','m_Pt_lep1', 'm_Pt_lep2','m_Pt_lep3','m_Pt_W', 'm_Pt_Z']]
        
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

        
def prepare_data(filedir, lumi):
    #Read background and signal files and save them as panda data frames
    
    #Background WZ
    bkgWZQCD = read_data(filedir+'364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root')
    
    #Background WZ EW
    bkgWZEW = read_data(filedir+'364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root')
    
    #Normalize weights and store these in panda data frames
    bkgWZQCD['Weight']=bkgWZQCD['Weight']*lumi*4583./5485580.
    bkgWZEW['Weight']=bkgWZEW['Weight']*lumi*47./1471000.

    bg=bkgWZQCD.append(bkgWZEW)
    #bg=bkgWZQCD
    bg['Label'] = '0'

    #Read signal
    sig300= read_data(filedir+'new/signalsNew/mc16d/305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_Systematics.root')
    sig400= read_data(filedir+'new/signalsNew/mc16d/305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_Systematics.root')
    sig500= read_data(filedir+'new/signalsNew/mc16d/305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_Systematics.root')
    sig600= read_data(filedir+'new/signalsNew/mc16d/305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_Systematics.root')
    sig700= read_data(filedir+'new/signalsNew/mc16d/305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_Systematics.root')
    sig800= read_data(filedir+'new/signalsNew/mc16d/305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_Systematics.root')
    sig900= read_data(filedir+'new/signalsNew/mc16d/305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_Systematics.root')

    sig300['Weight']=sig300['Weight']*lumi*3.9238/(2*40000.)
    sig300['LabelMass']=0
    sig400['Weight']=sig400['Weight']*lumi*2.4428/(2*40000.)
    sig400['LabelMass']=1
    sig500['Weight']=sig500['Weight']*lumi*1.6113/(2*40000.)
    sig500['LabelMass']=2
    sig600['Weight']=sig600['Weight']*lumi*1.1005/(2*40000.)
    sig600['LabelMass']=3
    sig700['Weight']=sig700['Weight']*lumi*0.7734/(2*40000.)
    sig700['LabelMass']=4
    sig800['Weight']=sig800['Weight']*lumi*0.55433/(2*40000.)
    sig800['LabelMass']=5
    sig900['Weight']=sig900['Weight']*lumi*0.40394/(2*40000.)
    sig900['LabelMass']=6

    sig=sig300.append([sig400,sig500,sig600,sig700,sig800,sig900])
    sig['Label'] = '1'

    #Probability distribution for random Mass Label
    prob=[float(sig300.shape[0])/sig.shape[0],float(sig400.shape[0])/sig.shape[0],float(sig500.shape[0])/sig.shape[0],float(sig600.shape[0])/sig.shape[0],float(sig700.shape[0])/sig.shape[0],float(sig800.shape[0])/sig.shape[0],float(sig900.shape[0])/sig.shape[0]]

    label=np.random.choice(7,bg.shape[0], p=prob)

    bg['LabelMass'] = label

    #Save prob distribution
    np.save('./prob', prob)
    
    data=bg.append(sig)
    data.loc[data.m_Valid_jet3 == 0, ['m_Eta_jet3','m_Y_jet3','m_Phi_jet3']] = -10., -10., -5.
    data = data.sample(frac=1).reset_index(drop=True)
    # Pick a random seed for reproducible results
    # Use 30% of the training sample for validation

    data_cont = dataset(data,1.,0.7)
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
    plt.savefig("NN_clf.png")
    plt.clf() 
