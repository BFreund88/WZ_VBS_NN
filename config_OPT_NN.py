import numpy as np
class input_samples:
    #Assumed luminosity
    lumi = 150.
    #Fraction reserved for testing 
    valfrac = 0.7
    #Directory where ntuples are located
    filedir = '/lcg/storage15/atlas/freund/ntuples_Miaoran/MVA_final/'
    #Bkg Samples
    bckgr = {
        'name' : ['MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root', 
                  'MVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root'],
        'xs' : [4583., 47.],
        'nevents' : [5485580, 1471000]
    }

    #Signal Samples
    filedirsig = filedir #+ 'new/signalsNew/mc16d/'
    sig = {
        'name' : ['MVA.305028_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_200_qcd0_ntuples.root',
                  'MVA.305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_ntuples.root',
                  'MVA.305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_ntuples.root',
                  'MVA.305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_ntuples.root',
                  'MVA.305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_ntuples.root',
                  'MVA.305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_ntuples.root',
                  'MVA.305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_ntuples.root',
                  'MVA.305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root'],
        'xs' : [7.06,3.9238,2.4428,1.6113,1.1005,0.7734,0.55433,0.40394],
        'nevents' : [40000,40000,40000,40000,40000,40000,40000,40000]
    }
    #Variables used for training
    variables = ['M_jj','Deta_jj','PtBalanceZ','PtBalanceW' ,'Jet1Pt', 'Jet2Pt', \
                 'Jet1Eta','Jet2Eta','Jet1E','Jet2E',\
                     'Lep1Eta','Lep2Eta', 'Lep3Eta','Lep1Pt', 'Lep2Pt','Lep3Pt','Pt_W',\
                     'Pt_Z','ZetaLep','Njets']
#No 3rd jet no MET 'MET'
#                     'm_Pt_jet3',,'m_Eta_jet3''m_E_jet3',

#Contains list of samples to apply NN
class apply_samples:
    filedirbkg = '/lcg/storage15/atlas/freund/ntuples_Miaoran/MVA_final/'
    list_apply_bkg = ['MVA.364253_Sherpa_222_NNPDF30NNLO_lllv_ntuples.root',
                      'MVA.364284_Sherpa_222_NNPDF30NNLO_lllvjj_EW6_ntuples.root']
    filedirsig = filedirbkg #+ 'new/signalsNew/mc16a/'
    list_apply_sig = ['MVA.305029_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_300_qcd0_ntuples.root',
                      'MVA.305030_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_400_qcd0_ntuples.root',
                      'MVA.305031_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_500_qcd0_ntuples.root',
                      'MVA.305032_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_600_qcd0_ntuples.root',
                      'MVA.305033_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_700_qcd0_ntuples.root',
                      'MVA.305034_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_800_qcd0_ntuples.root',
                      'MVA.305035_MGPy8_A14NNPDF30NLO_VBS_H5p_lvll_900_qcd0_ntuples.root']
    label = np.arange(len(list_apply_sig))
