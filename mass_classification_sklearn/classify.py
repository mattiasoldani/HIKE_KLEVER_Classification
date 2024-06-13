# single-file, zOptical/zWriteCSV-V3-CSV-based classification script for HTCondor; called by condor_process.sh which is executed within condor_run.sub

import sys
import numpy as np
import pickle
import sklearn

###

'''
arguments:
0) input data CSV file path+name
1) output prediction CSV file path+name
2) output histogram Pickle file path+name
3) input classifier #0 Pickle file path+name
4) input classifier #1 Pickle file path+name
''';

bSysArgs = True
if bSysArgs:
    filedataname = sys.argv[1]
    outpreddataname = sys.argv[2]
    outhistdataname = sys.argv[3]
    filecls0name = sys.argv[4]
    filecls1name = sys.argv[5]
else:
    with open('argsman.dat', 'r') as filedata:
        names_gen = filedata.readlines()
    filedataname = names_gen[0].replace("\n", "")
    outpreddataname = names_gen[1].replace("\n", "")
    outhistdataname = names_gen[2].replace("\n", "")
    filecls0name = names_gen[3].replace("\n", "")
    filecls1name = names_gen[4].replace("\n", "")
    
bV3 = False  # True for V3, False for V2
    
bShortBl = True  #(("SHORTBL" in filecls0name) | ("SHORTBL" in filecls1name))
    
bConv1Plus = False  #(("1+conv" in filecls0name) | ("1+conv" in filecls1name))
    
bBox = True if (("only_in_box_extended" in filecls0name) | ("only_in_box_extended" in filecls1name)) else False
shift_fv = 0 if bShortBl else 150
box = ((120-shift_fv, 190-shift_fv), (0.10, 100)) if bShortBl else ((250-shift_fv, 350-shift_fv), (0.10, 100))

if bV3:
    varnames = {
        "class" : 0,
        "iEv" : 1,
        "W" : 2,
        "iPair" : 3,
        "Vertex_nConverted" : 4,
        "Vertex_xRec_Z" : 5,
        "Vertex_pRecPi_T" : 6,
        "Vertex_xBest_T" : 7,
        "Clusters_xRec_TAsym" : 8,
        "Clusters_xDist" : 9,
        "Clusters_ECOG" : 10,
        "Clusters_EMin" : 11,
        "Clusters_ESum" : 12,
        "Vertex_x_ZDiffPreRec" : 13,
        "Clusters_ERatio" : 14,
        "Vertex_Best_EMinsTh" : 15,
        "Vertex_Best_EMaxsTh" : 16,
        "Vertex_pBest_dsPhi" : 17,
    }
else:
    varnames = {
        "class" : 0,
        "iEv" : 1,
        "W" : 2,
        "iPair" : 3,
        "Vertex_nConverted" : 4,
        "Vertex_xRec_Z" : 5,
        "Vertex_xRec_X" : 6,
        "Vertex_xRec_Y" : 7,
        "Vertex_xRecPre_Z" : 8,
        "Vertex_xRecPre_X" : 9,
        "Vertex_xRecPre_Y" : 10,
        "Cluster0_nHits" : 11,
        "Cluster0_xRec_Z" : 12,
        "Cluster0_xRec_X" : 13,
        "Cluster0_xRec_Y" : 14,
        "Vertex_pRec0_Z" : 15,
        "Vertex_pRec0_X" : 16,
        "Vertex_pRec0_Y" : 17,
        "Cluster0_ERec" : 18,
        "Cluster0_PosRes" : 19,
        "Cluster0_ERes" : 20,
        "Cluster0_xPre1_Z" : 21,
        "Cluster0_xPre1_X" : 22,
        "Cluster0_xPre1_Y" : 23,
        "Cluster0_xPre2_Z" : 24,
        "Cluster0_xPre2_X" : 25,
        "Cluster0_xPre2_Y" : 26,
        "Vertex_pRecPre0_Z" : 27,
        "Vertex_pRecPre0_X" : 28,
        "Vertex_pRecPre0_Y" : 29,
        "Cluster0_PreRes" : 30,
        "Cluster1_nHits" : 31,
        "Cluster1_xRec_Z" : 32,
        "Cluster1_xRec_X" : 33,
        "Cluster1_xRec_Y" : 34,
        "Vertex_pRec1_Z" : 35,
        "Vertex_pRec1_X" : 36,
        "Vertex_pRec1_Y" : 37,
        "Cluster1_ERec" : 38,
        "Cluster1_PosRes" : 39,
        "Cluster1_ERes" : 40,
        "Cluster1_xPre1_Z" : 41,
        "Cluster1_xPre1_X" : 42,
        "Cluster1_xPre1_Y" : 43,
        "Cluster1_xPre2_Z" : 44,
        "Cluster1_xPre2_X" : 45,
        "Cluster1_xPre2_Y" : 46,
        "Vertex_pRecPre1_Z" : 47,
        "Vertex_pRecPre1_X" : 48,
        "Vertex_pRecPre1_Y" : 49,
        "Cluster1_PreRes" : 50,
    }

###

# open data
df = np.loadtxt(filedataname, delimiter=",", skiprows=1)
df = df[~np.isnan(df).any(axis=1)]
if bBox:
    if bV3:
        df = df[
            ((df.T[varnames["Vertex_xRec_Z"]]>box[0][0]) &\
            (df.T[varnames["Vertex_xRec_Z"]]<box[0][1])) &\
            ((df.T[varnames["Vertex_pRecPi_T"]]>box[1][0]) &\
            (df.T[varnames["Vertex_pRecPi_T"]]<box[1][1]))
        ]
    else:
        df = df[
            ((df.T[varnames["Vertex_xRec_Z"]]>box[0][0]) &\
            (df.T[varnames["Vertex_xRec_Z"]]<box[0][1])) &\
            ((np.sqrt( (df.T["Vertex_pRec0_X"] + df.T["Vertex_pRec1_X"])**2 + (df.T["Vertex_pRec0_Y"] + df.T["Vertex_pRec1_Y"])**2 )>box[1][0]) &\
            (np.sqrt( (df.T["Vertex_pRec0_X"] + df.T["Vertex_pRec1_X"])**2 + (df.T["Vertex_pRec0_Y"] + df.T["Vertex_pRec1_Y"])**2 )<box[1][1]))
        ]
if bConv1Plus:
    df = df[df.T[varnames["Vertex_nConverted"]]>0]
df = df.T

print("data loaded")

# depending on the file format, condition data
# vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

if not bV3:

    # select best vertex quantities
    varnames["Vertex_xBest_Z"] = len(varnames.keys())
    varnames["Vertex_xBest_X"] = len(varnames.keys())
    varnames["Vertex_xBest_Y"] = len(varnames.keys())
    varnames["Vertex_pBest0_Z"] = len(varnames.keys())
    varnames["Vertex_pBest0_X"] = len(varnames.keys())
    varnames["Vertex_pBest0_Y"] = len(varnames.keys())
    varnames["Vertex_pBest1_Z"] = len(varnames.keys())
    varnames["Vertex_pBest1_X"] = len(varnames.keys())
    varnames["Vertex_pBest1_Y"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_xRec_Z"]], df[varnames["Vertex_xRecPre_Z"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_xRec_X"]], df[varnames["Vertex_xRecPre_X"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_xRec_Y"]], df[varnames["Vertex_xRecPre_Y"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec0_Z"]], df[varnames["Vertex_pRecPre0_Z"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec0_X"]], df[varnames["Vertex_pRecPre0_X"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec0_Y"]], df[varnames["Vertex_pRecPre0_Y"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec1_Z"]], df[varnames["Vertex_pRecPre1_Z"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec1_X"]], df[varnames["Vertex_pRecPre1_X"]]) ].T 
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, df[varnames["Vertex_pRec1_Y"]], df[varnames["Vertex_pRecPre1_Y"]]) ].T 
    
    # compute all transverse quantities from cartesian components
    varnames["Vertex_xRec_T"] = len(varnames.keys())
    varnames["Vertex_xRecPre_T"] = len(varnames.keys())
    varnames["Cluster0_xRec_T"] = len(varnames.keys())
    varnames["Cluster1_xRec_T"] = len(varnames.keys())
    varnames["Vertex_pRec0_T"] = len(varnames.keys())
    varnames["Vertex_pRec1_T"] = len(varnames.keys())
    varnames["Cluster0_xPre1_T"] = len(varnames.keys())
    varnames["Cluster0_xPre2_T"] = len(varnames.keys())
    varnames["Cluster1_xPre1_T"] = len(varnames.keys())
    varnames["Cluster1_xPre2_T"] = len(varnames.keys())
    varnames["Vertex_pRecPre0_T"] = len(varnames.keys())
    varnames["Vertex_pRecPre1_T"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_xRec_X"]]**2 + df[varnames["Vertex_xRec_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_xRecPre_X"]]**2 + df[varnames["Vertex_xRecPre_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster0_xRec_X"]]**2 + df[varnames["Cluster0_xRec_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster1_xRec_X"]]**2 + df[varnames["Cluster1_xRec_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRec0_X"]]**2 + df[varnames["Vertex_pRec0_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRec1_X"]]**2 + df[varnames["Vertex_pRec1_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster0_xPre1_X"]]**2 + df[varnames["Cluster0_xPre1_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster0_xPre2_X"]]**2 + df[varnames["Cluster0_xPre2_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster1_xPre1_X"]]**2 + df[varnames["Cluster1_xPre1_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Cluster1_xPre2_X"]]**2 + df[varnames["Cluster1_xPre2_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRecPre0_X"]]**2 + df[varnames["Vertex_pRecPre0_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRecPre1_X"]]**2 + df[varnames["Vertex_pRecPre1_Y"]]**2 ) ].T

    varnames["Vertex_xBest_T"] = len(varnames.keys())
    varnames["Vertex_pBest0_T"] = len(varnames.keys())
    varnames["Vertex_pBest1_T"] = len(varnames.keys())   
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_xBest_X"]]**2 + df[varnames["Vertex_xBest_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pBest0_X"]]**2 + df[varnames["Vertex_pBest0_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pBest1_X"]]**2 + df[varnames["Vertex_pBest1_Y"]]**2 ) ].T
    
    # compute new 2-cluster position and momentum-related variables
    varnames["Clusters_xRec_TMin"] = len(varnames.keys())
    varnames["Clusters_xPre1_TMin"] = len(varnames.keys())
    varnames["Clusters_xPre2_TMin"] = len(varnames.keys())
    varnames["Vertex_pRec_TMin"] = len(varnames.keys())
    varnames["Vertex_pRecPre_TMin"] = len(varnames.keys())
    varnames["Clusters_xRec_TMax"] = len(varnames.keys())
    varnames["Clusters_xPre1_TMax"] = len(varnames.keys())
    varnames["Clusters_xPre2_TMax"] = len(varnames.keys())
    varnames["Vertex_pRec_TMax"] = len(varnames.keys())
    varnames["Vertex_pRecPre_TMax"] = len(varnames.keys())
    varnames["Clusters_xRec_TSum"] = len(varnames.keys())
    varnames["Clusters_xPre1_TSum"] = len(varnames.keys())
    varnames["Clusters_xPre2_TSum"] = len(varnames.keys())
    varnames["Vertex_pRec_TSum"] = len(varnames.keys())
    varnames["Vertex_pRecPre_TSum"] = len(varnames.keys())
    varnames["Clusters_xRec_TDif"] = len(varnames.keys())
    varnames["Clusters_xPre1_TDif"] = len(varnames.keys())
    varnames["Clusters_xPre2_TDif"] = len(varnames.keys())
    varnames["Vertex_pRec_TDif"] = len(varnames.keys())
    varnames["Vertex_pRecPre_TDif"] = len(varnames.keys())
    varnames["Clusters_xRec_TAsym"] = len(varnames.keys())
    varnames["Clusters_xPre1_TAsym"] = len(varnames.keys())
    varnames["Clusters_xPre2_TAsym"] = len(varnames.keys())
    varnames["Vertex_pRec_TAsym"] = len(varnames.keys())
    varnames["Vertex_pRecPre_TAsym"] = len(varnames.keys())    
    df = np.c_[ df.T, np.minimum(df[varnames["Cluster0_xRec_T"]], df[varnames["Cluster1_xRec_T"]]) ].T    
    df = np.c_[ df.T, np.minimum(df[varnames["Cluster0_xPre1_T"]], df[varnames["Cluster1_xPre1_T"]]) ].T       
    df = np.c_[ df.T, np.minimum(df[varnames["Cluster0_xPre2_T"]], df[varnames["Cluster1_xPre2_T"]]) ].T       
    df = np.c_[ df.T, np.minimum(df[varnames["Vertex_pRec0_T"]], df[varnames["Vertex_pRec1_T"]]) ].T       
    df = np.c_[ df.T, np.minimum(df[varnames["Vertex_pRecPre0_T"]], df[varnames["Vertex_pRecPre1_T"]]) ].T       
    df = np.c_[ df.T, np.maximum(df[varnames["Cluster0_xRec_T"]], df[varnames["Cluster1_xRec_T"]]) ].T       
    df = np.c_[ df.T, np.maximum(df[varnames["Cluster0_xPre1_T"]], df[varnames["Cluster1_xPre1_T"]]) ].T       
    df = np.c_[ df.T, np.maximum(df[varnames["Cluster0_xPre2_T"]], df[varnames["Cluster1_xPre2_T"]]) ].T       
    df = np.c_[ df.T, np.maximum(df[varnames["Vertex_pRec0_T"]], df[varnames["Vertex_pRec1_T"]]) ].T       
    df = np.c_[ df.T, np.maximum(df[varnames["Vertex_pRecPre0_T"]], df[varnames["Vertex_pRecPre1_T"]]) ].T      
    df = np.c_[ df.T, df[varnames["Clusters_xRec_TMax"]] + df[varnames["Clusters_xRec_TMin"]] ].T
    df = np.c_[ df.T, df[varnames["Clusters_xPre1_TMax"]] + df[varnames["Clusters_xPre1_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xPre2_TMax"]] + df[varnames["Clusters_xPre2_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRec_TMax"]] + df[varnames["Vertex_pRec_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre_TMax"]] + df[varnames["Vertex_pRecPre_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xRec_TMax"]] - df[varnames["Clusters_xRec_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xPre1_TMax"]] - df[varnames["Clusters_xPre1_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xPre2_TMax"]] - df[varnames["Clusters_xPre2_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRec_TMax"]] - df[varnames["Vertex_pRec_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre_TMax"]] - df[varnames["Vertex_pRecPre_TMin"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xRec_TDif"]] / df[varnames["Clusters_xRec_TSum"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xPre1_TDif"]] / df[varnames["Clusters_xPre1_TSum"]] ].T 
    df = np.c_[ df.T, df[varnames["Clusters_xPre2_TDif"]] / df[varnames["Clusters_xPre2_TSum"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRec_TDif"]] / df[varnames["Vertex_pRec_TSum"]] ].T 
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre_TDif"]] / df[varnames["Vertex_pRecPre_TSum"]] ].T 

    varnames["Vertex_pBest_TMin"] = len(varnames.keys())
    varnames["Vertex_pBest_TMax"] = len(varnames.keys())
    varnames["Vertex_pBest_TSum"] = len(varnames.keys())
    varnames["Vertex_pBest_TDif"] = len(varnames.keys())
    varnames["Vertex_pBest_TAsym"] = len(varnames.keys())
    df = np.c_[ df.T, np.minimum( df[varnames["Vertex_pBest0_T"]], df[varnames["Vertex_pBest1_T"]] ) ].T
    df = np.c_[ df.T, np.maximum( df[varnames["Vertex_pBest0_T"]], df[varnames["Vertex_pBest1_T"]] ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest_TMax"]] + df[varnames["Vertex_pBest_TMin"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest_TMax"]] - df[varnames["Vertex_pBest_TMin"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest_TDif"]] / df[varnames["Vertex_pBest_TSum"]] ].T

    varnames["Clusters_xDist"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt(  # distance between clusters
        (df[varnames["Cluster1_xRec_X"]]-df[varnames["Cluster0_xRec_X"]])**2 + (df[varnames["Cluster1_xRec_Y"]]-df[varnames["Cluster0_xRec_Y"]])**2
    ) ].T

    # compute new 2-cluster energy-related variables
    varnames["Clusters_EMin"] = len(varnames.keys())
    varnames["Clusters_EMax"] = len(varnames.keys())
    varnames["Clusters_ESum"] = len(varnames.keys())
    varnames["Clusters_EDif"] = len(varnames.keys())
    varnames["Clusters_EAsym"] = len(varnames.keys())
    df = np.c_[ df.T, np.minimum( df[varnames["Cluster0_ERec"]], df[varnames["Cluster1_ERec"]] ) ].T
    df = np.c_[ df.T, np.maximum( df[varnames["Cluster0_ERec"]], df[varnames["Cluster1_ERec"]] ) ].T
    df = np.c_[ df.T, df[varnames["Clusters_EMax"]] + df[varnames["Clusters_EMin"]] ].T
    df = np.c_[ df.T, df[varnames["Clusters_EMax"]] - df[varnames["Clusters_EMin"]] ].T
    df = np.c_[ df.T, df[varnames["Clusters_EDif"]] / df[varnames["Clusters_ESum"]] ].T

    varnames["Clusters_xRec_TEMin"] = len(varnames.keys())
    varnames["Clusters_xRec_TEMax"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Cluster0_xRec_T"]], df[varnames["Cluster1_xRec_T"]]) ].T
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Cluster1_xRec_T"]], df[varnames["Cluster0_xRec_T"]]) ].T

    varnames["Clusters_ECOG"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt(  # cog. between clusters (weighted with energies)
        (
            (df[varnames["Cluster0_ERec"]]*df[varnames["Cluster0_xRec_X"]] + df[varnames["Cluster1_ERec"]]*df[varnames["Cluster1_xRec_X"]])**2 +\
            (df[varnames["Cluster0_ERec"]]*df[varnames["Cluster0_xRec_Y"]] + df[varnames["Cluster1_ERec"]]*df[varnames["Cluster1_xRec_Y"]])**2
        ) / df[varnames["Clusters_ESum"]]**2
    ) ].T

    # compute pion kinematics from two photons (pion energy is simply Clusters_ESum)
    varnames["Vertex_pRecPi_Z"] = len(varnames.keys())
    varnames["Vertex_pRecPi_X"] = len(varnames.keys())
    varnames["Vertex_pRecPi_Y"] = len(varnames.keys())
    varnames["Vertex_pRecPrePi_Z"] = len(varnames.keys())
    varnames["Vertex_pRecPrePi_X"] = len(varnames.keys())
    varnames["Vertex_pRecPrePi_Y"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pRec0_Z"]] + df[varnames["Vertex_pRec1_Z"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRec0_X"]] + df[varnames["Vertex_pRec1_X"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRec0_Y"]] + df[varnames["Vertex_pRec1_Y"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre0_Z"]] + df[varnames["Vertex_pRecPre1_Z"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre0_X"]] + df[varnames["Vertex_pRecPre1_X"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre0_Y"]] + df[varnames["Vertex_pRecPre1_Y"]] ].T

    varnames["Vertex_pRecPi_T"] = len(varnames.keys())
    varnames["Vertex_pRecPrePi_T"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRecPi_X"]]**2 + df[varnames["Vertex_pRecPi_Y"]]**2 ) ].T
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pRecPrePi_X"]]**2 + df[varnames["Vertex_pRecPrePi_Y"]]**2 ) ].T

    varnames["Vertex_pRecPi_sTh"] = len(varnames.keys())
    varnames["Vertex_pRecPrePi_sTh"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pRecPi_T"]] / np.sqrt(  # pion propagation angle wrt. beam
        df[varnames["Vertex_pRecPi_T"]]**2 + df[varnames["Vertex_pRecPi_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPrePi_T"]] / np.sqrt(  # pion propagation angle wrt. beam
        df[varnames["Vertex_pRecPrePi_T"]]**2 + df[varnames["Vertex_pRecPrePi_Z"]]**2
    ) ].T

    varnames["Vertex_pBestPi_Z"] = len(varnames.keys())
    varnames["Vertex_pBestPi_X"] = len(varnames.keys())
    varnames["Vertex_pBestPi_Y"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pBest0_Z"]] + df[varnames["Vertex_pBest1_Z"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest0_X"]] + df[varnames["Vertex_pBest1_X"]] ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest0_Y"]] + df[varnames["Vertex_pBest1_Y"]] ].T

    varnames["Vertex_pBestPi_T"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt( df[varnames["Vertex_pBestPi_X"]]**2 + df[varnames["Vertex_pBestPi_Y"]]**2 ) ].T

    varnames["Vertex_pBestPi_sTh"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pBestPi_T"]] / np.sqrt(  # pion propagation angle wrt. beam
        df[varnames["Vertex_pBestPi_T"]]**2 + df[varnames["Vertex_pBestPi_Z"]]**2
    ) ].T

    # new ideas
    varnames["Vertex_x_ZDiffPreRec"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, 9999, df[varnames["Vertex_xRec_Z"]]-df[varnames["Vertex_xRecPre_Z"]]) ].T
    # ^^^ Matt: diff. between vertex position from MEC only and from MEC+PSD, longitudinal

    varnames["Vertex_x_TDiffPreRec"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, 9999, df[varnames["Vertex_xRec_T"]]-df[varnames["Vertex_xRecPre_T"]]) ].T
    # ^^^ Matt: diff. between vertex position from MEC only and from MEC+PSD, transverse

    varnames["Vertex_x_DistDiffPreRec"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Vertex_nConverted"]]==0, 9999, np.sqrt(  
        (df[varnames["Vertex_xRec_X"]] - df[varnames["Vertex_xRecPre_X"]])**2 +\
        (df[varnames["Vertex_xRec_Y"]] - df[varnames["Vertex_xRecPre_Y"]])**2 +\
        (df[varnames["Vertex_xRec_Z"]] - df[varnames["Vertex_xRecPre_Z"]])**2
    )) ].T
    # ^^^ Matt: diff. between vertex position from MEC only and from MEC+PSD, total

    varnames["Clusters_ERatio"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Clusters_EMin"]] / df[varnames["Clusters_EMax"]] ].T 
    # ^^^ from KOTO - lowest-to-highest energy ratio

    varnames["Vertex_pRec0_sTh"] = len(varnames.keys())
    varnames["Vertex_pRec1_sTh"] = len(varnames.keys())
    varnames["Vertex_pRecPre0_sTh"] = len(varnames.keys())
    varnames["Vertex_pRecPre1_sTh"] = len(varnames.keys())
    varnames["Vertex_Rec0_EsTh"] = len(varnames.keys())
    varnames["Vertex_Rec1_EsTh"] = len(varnames.keys())
    varnames["Vertex_RecPre0_EsTh"] = len(varnames.keys())
    varnames["Vertex_RecPre1_EsTh"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pRec0_T"]] / np.sqrt(
        df[varnames["Vertex_pRec0_T"]]**2 + df[varnames["Vertex_pRec0_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRec1_T"]] / np.sqrt(
        df[varnames["Vertex_pRec1_T"]]**2 + df[varnames["Vertex_pRec1_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre0_T"]] / np.sqrt(
        df[varnames["Vertex_pRecPre0_T"]]**2 + df[varnames["Vertex_pRecPre0_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPre1_T"]] / np.sqrt(
        df[varnames["Vertex_pRecPre1_T"]]**2 + df[varnames["Vertex_pRecPre1_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Cluster0_ERec"]]*df[varnames["Vertex_pRec0_sTh"]] ].T
    df = np.c_[ df.T, df[varnames["Cluster1_ERec"]]*df[varnames["Vertex_pRec1_sTh"]] ].T
    df = np.c_[ df.T, df[varnames["Cluster0_ERec"]]*df[varnames["Vertex_pRecPre0_sTh"]] ].T
    df = np.c_[ df.T, df[varnames["Cluster1_ERec"]]*df[varnames["Vertex_pRecPre1_sTh"]] ].T
    # ^^^ KOTO - single-photon angle*energy 

    varnames["Vertex_Rec_EMinsTh"] = len(varnames.keys())
    varnames["Vertex_Rec_EMaxsTh"] = len(varnames.keys())
    varnames["Vertex_RecPre_EMinsTh"] = len(varnames.keys())
    varnames["Vertex_RecPre_EMaxsTh"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_Rec0_EsTh"]], df[varnames["Vertex_Rec1_EsTh"]]) ].T
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_Rec1_EsTh"]], df[varnames["Vertex_Rec0_EsTh"]]) ].T
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_RecPre0_EsTh"]], df[varnames["Vertex_RecPre1_EsTh"]]) ].T
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_RecPre1_EsTh"]], df[varnames["Vertex_RecPre0_EsTh"]]) ].T
    # ^^^ KOTO - single-photon angle*energy 

    varnames["Vertex_pBest0_sTh"] = len(varnames.keys())
    varnames["Vertex_pBest1_sTh"] = len(varnames.keys())
    varnames["Vertex_Best0_EsTh"] = len(varnames.keys())
    varnames["Vertex_Best1_EsTh"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pBest0_T"]] / np.sqrt(
        df[varnames["Vertex_pBest0_T"]]**2 + df[varnames["Vertex_pBest0_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pBest1_T"]] / np.sqrt(
        df[varnames["Vertex_pBest1_T"]]**2 + df[varnames["Vertex_pBest1_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Cluster0_ERec"]]*df[varnames["Vertex_pBest0_sTh"]] ].T
    df = np.c_[ df.T, df[varnames["Cluster1_ERec"]]*df[varnames["Vertex_pBest1_sTh"]] ].T
    # ^^^ KOTO - single-photon angle*energy 

    varnames["Vertex_Best_EMinsTh"] = len(varnames.keys())
    varnames["Vertex_Best_EMaxsTh"] = len(varnames.keys())
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_Best0_EsTh"]], df[varnames["Vertex_Best1_EsTh"]]) ].T
    df = np.c_[ df.T, np.where(df[varnames["Cluster0_ERec"]]==df[varnames["Clusters_EMin"]], df[varnames["Vertex_Best1_EsTh"]], df[varnames["Vertex_Best0_EsTh"]]) ].T
    # ^^^ KOTO - single-photon angle*energy 

    varnames["Vertex_pRec0_sPhi"] = len(varnames.keys())
    varnames["Vertex_pRec1_sPhi"] = len(varnames.keys())
    varnames["Vertex_pRecPre0_sPhi"] = len(varnames.keys())
    varnames["Vertex_pRecPre1_sPhi"] = len(varnames.keys())
    varnames["Vertex_pRec_dsPhi"] = len(varnames.keys())
    varnames["Vertex_pRecPre_dsPhi"] = len(varnames.keys())
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pRec0_Y"]] / df[varnames["Vertex_pRec0_T"]]) ].T
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pRec1_Y"]] / df[varnames["Vertex_pRec1_T"]]) ].T
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pRecPre0_Y"]] / df[varnames["Vertex_pRecPre0_T"]]) ].T
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pRecPre1_Y"]] / df[varnames["Vertex_pRecPre1_T"]]) ].T
    df = np.c_[ df.T, np.abs(df[varnames["Vertex_pRec1_sPhi"]] - df[varnames["Vertex_pRec0_sPhi"]]) ].T
    df = np.c_[ df.T, np.abs(df[varnames["Vertex_pRecPre1_sPhi"]] - df[varnames["Vertex_pRecPre0_sPhi"]]) ].T
    # ^^^ KOTO - transv. angle between photons
        
    varnames["Vertex_pBest0_sPhi"] = len(varnames.keys())
    varnames["Vertex_pBest1_sPhi"] = len(varnames.keys())
    varnames["Vertex_pBest_dsPhi"] = len(varnames.keys())
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pBest0_Y"]] / df[varnames["Vertex_pBest0_T"]]) ].T
    df = np.c_[ df.T, np.arcsin(df[varnames["Vertex_pBest1_Y"]] / df[varnames["Vertex_pBest1_T"]]) ].T
    df = np.c_[ df.T, np.abs(df[varnames["Vertex_pBest1_sPhi"]] - df[varnames["Vertex_pBest0_sPhi"]]) ].T
    # ^^^ KOTO - transv. angle between photons
    
    mPi = 134.9768e-3  # pion mass in GeV
    
    varnames["Vertex_ERecPi"] = len(varnames.keys())
    varnames["Vertex_ERecPrePi"] = len(varnames.keys())
    df = np.c_[ df.T, np.sqrt(df[varnames["Vertex_pRecPi_Z"]]**2 + df[varnames["Vertex_pRecPi_T"]]**2 + mPi**2) ].T
    df = np.c_[ df.T, np.sqrt(df[varnames["Vertex_pRecPrePi_Z"]]**2 + df[varnames["Vertex_pRecPrePi_T"]]**2 + mPi**2) ].T
    # ^^^ new variables for mass classification check - pion energy
    
    varnames["Vertex_RecPi_sTh"] = len(varnames.keys())
    varnames["Vertex_RecPrePi_sTh"] = len(varnames.keys())
    df = np.c_[ df.T, df[varnames["Vertex_pRecPi_T"]] / np.sqrt(
        df[varnames["Vertex_pRecPi_T"]]**2 + df[varnames["Vertex_pRecPi_Z"]]**2
    ) ].T
    df = np.c_[ df.T, df[varnames["Vertex_pRecPrePi_T"]] / np.sqrt(
        df[varnames["Vertex_pRecPrePi_T"]]**2 + df[varnames["Vertex_pRecPrePi_Z"]]**2
    ) ].T
    # ^^^ new variables for mass classification check - pion angle

# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

varnamescls = [
    varnames["Vertex_xBest_T"],
    varnames["Clusters_xRec_TAsym"],
    varnames["Clusters_xDist"],
    varnames["Clusters_ECOG"],
    varnames["Clusters_EMin"],
    varnames["Clusters_ESum"],
    varnames["Vertex_x_ZDiffPreRec"],
    varnames["Clusters_ERatio"],
    varnames["Vertex_Best_EMinsTh"],
    varnames["Vertex_Best_EMaxsTh"],
    varnames["Vertex_pBest_dsPhi"],
] 

# open classifiers
clss = []
fileclsnames = [filecls0name, filecls1name]
for fileclsname in fileclsnames:
    with open(fileclsname, 'rb') as filecls:
        clstemp = pickle.load(filecls)
    clss.append(clstemp if type(clstemp)!=type(()) else clstemp[0])
print("classifiers loaded")
    
###
    
# classification
X = df[varnamescls]
y_preds_float = []
y_preds_int = []
for i, cls in enumerate(clss):
    y_pred_float_temp = cls["classifier"].predict_proba(X.T)[:, np.where(cls["classifier"].classes_==1)[0][0]]
    cut = cls["output_cut"][cls["output_cut"]["used_for_evaluation"]]
    #cut = 0.5018
    #cut = 0.5030
    #cut = 0.5027
    #cut = 0.5062 if i==0 else cls["output_cut"][cls["output_cut"]["used_for_evaluation"]]
    #cut = 0.5027 if i==0 else cls["output_cut"][cls["output_cut"]["used_for_evaluation"]]
    #cut = 0.5032 if i==0 else cls["output_cut"][cls["output_cut"]["used_for_evaluation"]]
    #cut = 0.5030 if i==0 else cls["output_cut"][cls["output_cut"]["used_for_evaluation"]]
    #cut = 0.5030 if i==0 else 0.5150
    #cut = 0.5030 if i==0 else 0.5075
    #cut = 0.5030 if i==0 else 0.5100
    y_preds_float.append(y_pred_float_temp)
    y_preds_int.append((y_pred_float_temp > cut).astype('int'))
print("classification done")
    
###

# save predictions
df_out = df[varnames["iEv"]].astype("int")
header_out = "iEv"
df_out = np.vstack((df_out, df[varnames["W"]].astype("float")))
header_out += (",W")
df_out = np.vstack((df_out, df[varnames["Vertex_xRec_Z"]].astype("float")))
header_out += (",Vertex_xRec_Z")
df_out = np.vstack((df_out, df[varnames["Vertex_pRecPi_T"]].astype("float")))
header_out += (",Vertex_pRecPi_T")
for i in range(len(clss)):
    df_out = np.vstack((df_out, np.array(y_preds_float[i])))
    header_out += (",pred%d" % i)
df_out = np.vstack((df_out, df[varnames["Vertex_ERecPi"]].astype("float")))
header_out += (",Vertex_ERecPi")
df_out = np.vstack((df_out, df[varnames["Vertex_RecPi_sTh"]].astype("float")))
header_out += (",Vertex_RecPi_sTh")
df_out = np.vstack((df_out, df[varnames["Clusters_xRec_TMin"]].astype("float")))
header_out += (",Clusters_xRec_TMin")
np.savetxt(outpreddataname, df_out.T, delimiter=",", header=header_out)
print("predictions saved")

####
#
## save histograms
## 1st index ("hist_0/1") identifies the BDT, 2nd index (0/1) the predicted class
#bins = (200, 100)
#range_plot = ((0, 250), (0, 0.4)) 
#hist = {}
#for i, cls in enumerate(clss):
#    hist["hist_%d" % i] = {}
#    bool_sig = y_preds_int[i].astype("bool")
#    hist["hist_%d" % i][0] = np.histogram2d(
#        df[varnames["Vertex_xRec_Z"]][~bool_sig], df[varnames["Vertex_pRecPi_T"]][~bool_sig],
#        bins=bins, range=range_plot, weights=df[varnames["W"]][~bool_sig] if i==1 else None,
#    )
#    hist["hist_%d" % i][1] = np.histogram2d(
#        df[varnames["Vertex_xRec_Z"]][bool_sig], df[varnames["Vertex_pRecPi_T"]][bool_sig],
#        bins=bins, range=range_plot, weights=df[varnames["W"]][bool_sig] if i==1 else None,
#    )
#with open(outhistdataname, "wb") as filehist:
#    pickle.dump(hist, filehist)
#print("histograms saved")