# HIKE_KLEVER_Classification

This is a set of preliminary studies performed in 2023 on the signal/background classification performance at the CERN HIKE (High Intensity Kaon Experiments) experiment, KLEVER (K-Long Experiment for VEry Rare events) phase &mdash; details can be found e.g. in [this document](https://arxiv.org/abs/2211.16586). These studies were performed on datasets generated with the [zOptical](https://gitlab.cern.ch/prin-klever/zOptical) MC simulation software. In particular, events with two photons reconstructed in the Main Electromagnetic Calorimeter (and optionally in the PreShower Detector) and no other hits anywhere in the detector are considered from three decay channels:

- K-long into a pi-0 and two neutrino (PNN, BR~3e-11) &mdash; the channel of interest;
- K-long into two pi-0 (BR~0.864e-3), with two of the four resulting photons leaving the detector undetected;
- lambda-0 into a pi-0 and a neutron (BR~0.358).

The repository includes these tools:

- `mass_analysis_boolean` for the cut-based data analysis, mostly optimised for execution on an [Apache Spark](https://spark.apache.org/) cluster.
- `classifiers_sklearn` to train ML classifiers on the KLEVER datasets (of limited size) &mdash; in 2023, boosted decision trees were tested.
- `trained_algorithms` which contains Pickle files with the trained BDTs that were used in 2023, together with some basic performance plots.
- `mass_classification_sklearn`, a software to be run on [HTCondor](https://htcondor.org/) to perform the ML classification on full-scale datasets.
- `mass_final_analysis`, a set of analysis tools for the `mass_classification_sklearn` analysis output data.

Some of the main results of this work can be found e.g. in [this meeting contribution](https://indico.cern.ch/event/1234203/contributions/5560870/).

Basic dependencies of the Python code and Jupyter notebooks:

[![jupyterlab](https://img.shields.io/badge/jupyterlab-grey.svg)](https://jupyterlab.readthedocs.io/en/stable/)
[![matplotlib](https://img.shields.io/badge/matplotlib-grey.svg)](https://matplotlib.org/)
[![numpy](https://img.shields.io/badge/numpy-grey.svg)](https://numpy.org/)
[![pandas](https://img.shields.io/badge/pandas-grey.svg)](https://pandas.pydata.org/)
[![pip](https://img.shields.io/badge/pip-grey.svg)](https://pip.pypa.io/en/stable/)
[![pyspark](https://img.shields.io/badge/pyspark-grey.svg)](https://spark.apache.org/docs/latest/api/python/index.html#)
[![seaborn](https://img.shields.io/badge/seaborn-grey.svg)](https://seaborn.pydata.org/)
[![sklearn](https://img.shields.io/badge/sklearn-grey.svg)](https://scikit-learn.org/stable/)
[![succolib](https://img.shields.io/badge/succolib-grey.svg)](https://github.com/mattiasoldani/succolib)
