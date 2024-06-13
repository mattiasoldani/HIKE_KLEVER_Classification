# creates condor_args.dat required for condor_run.sub

import os

maindirexec="/SW_MASTER_PATH/mass_classification_sklearn/"

maindirdata="/DATA_MASTER_PATH/23_hike_pinunu-background/2309_zoptical-zanalyze_final_vars_SHORTBL/raw_csv_V2/"
#maindirdata="/DATA_MASTER_PATH/23_hike_pinunu-background/2305_zoptical-zanalyze_training/"
#maindirdata="/DATA_MASTER_PATH/23_hike_pinunu-background/2306_zoptical-zanalyze_lambda_mass_prod/"
#maindirdata="/DATA_MASTER_PATH/23_hike_pinunu-background/2305_zoptical-zanalyze_2pi_mass_prod/"

maindircls="/SW_MASTER_PATH/classifiers_sklearn/output_misc/" 

def filesel(s):
    #sel = ("sig_mal" in s) | ("bkg_2p0_mal" in s)
    #sel = ("sig_mal" in s) | ("lambda" in s)
    #sel = ("bkg_2p0_mal" in s)
    sel = True
    return sel

files = [s for s in os.listdir(maindirdata) if ((os.path.isfile(os.path.join(maindirdata, s))) & filesel(s))]

filecls0name = maindircls + "signal_vs_2pi_normalised/bdt_ab.pickle"
filecls1name = maindircls + "signal_vs_lambda_normalised/bdt_ab.pickle"
with open('condor_args.dat', 'w') as f:
    for i, s in enumerate(files):
        filedataname = maindirdata + s
        outpreddataname = maindirexec + "output_pred/" + s
        outhistdataname = maindirexec + "output_hist/" + s.replace(".csv", ".pickle")
        eol = "\n" if i<(len(files)-1) else ""
        f.write("%s %s %s %s %s %s%s" % (maindirexec, filedataname, outpreddataname, outhistdataname, filecls0name, filecls1name, eol))
