import numpy as np
class input_samples:
    #Assumed luminosity
    lumi = 150.
    #Fraction reserved for testing 
    valfrac = 0.7
    #Directory where ntuples are located
    filedir = '/lcg/storage15/atlas/freund/ntuples_Miaoran/'
    #Bkg Samples
    bckgr = {
        'name' : ['364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root', 
                  '364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root'],
        'xs' : [4583., 47.],
        'nevents' : [5485580, 1471000]
    }

    #Signal Samples
    filedirsig = filedir + 'new/signalsNew/mc16d/'
    sig = {
        'name' : ['305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_Systematics.root',
                  '305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_Systematics.root',
                  '305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_Systematics.root',
                  '305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_Systematics.root',
                  '305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_Systematics.root',
                  '305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_Systematics.root',
                  '305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_Systematics.root'],
        'xs' : [3.9238,2.4428,1.6113,1.1005,0.7734,0.55433,0.40394],
        'nevents' : [40000,40000,40000,40000,40000,40000,40000]
    }
    #Variables used for training
    variables = ['Mjj','Detajj','MET','PtZoverWZmass','PtWoverWZmass' ,'m_Pt_jet1','m_Pt_jet2', \
                     'm_Pt_jet3','m_Eta_jet1','m_Eta_jet2','m_Eta_jet3','m_E_jet1','m_E_jet2','m_E_jet3',\
                     'm_Eta_lep1','m_Eta_lep2', 'm_Eta_lep3','m_Pt_lep1', 'm_Pt_lep2','m_Pt_lep3','m_Pt_W',\
                     'm_Pt_Z']

#Contains list of samples to apply NN
class apply_samples:
    filedirbkg = '/lcg/storage15/atlas/freund/ntuples_Miaoran/'
    list_apply_bkg = ['364253_Sherpa_222_NNPDF30NNLO_lllv_Systematics.root',
                      '364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_Systematics.root']
    filedirsig = filedirbkg + 'new/signalsNew/mc16a/'
    list_apply_sig = ['305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_Systematics.root',
                      '305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_Systematics.root',
                      '305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_Systematics.root',
                      '305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_Systematics.root',
                      '305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_Systematics.root',
                      '305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_Systematics.root',
                      '305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_Systematics.root']
    label = np.arange(len(list_apply_sig))
