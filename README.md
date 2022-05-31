# BRCA_CART_model
This program performs stratified 10-fold Cross-validation (CV) and Stratified 10X10 nested-CV) to evaluate optimized predictive models based on datasets derived from the TCGA-BRCA project (https://gdc.cancer.gov/).

To run the code examples, python version 3.8.5 is required, which can be downloaded from https://www.python.org/downloads/?fbclid=IwAR1VrxINLd7djn_4UuQTxshpya9D2wmt_H0jnUaJrEXVCoalD3viphyLRyI

Python Libraries used in this study are listed below and can be installed using Python’s in-build pip module by executing the code directly from Linux terminal command line.

    Pandas version 1.2.3
    Scikit-learn version 0.20.1
    Imbalanced-learn version 0.3.3
    Dtreeviz version 1.2
    Joblib version 1.0.1
    Graphviz version 0.16
    Matplotlib version 3.3.4
    Numpy version 1.20.2
    Seaborn version 0.11.1

Code example shoud be executed directly in the working directory from the Linux terminal command line

Usage: CART.py [-h] -I DATASET -c CLASSIFIER [-p HYPERPARAM] [-r NB_RANDOM_SEEDS] [-m METRIC] [-SW] [-log2] [-imp] [-t]

Optional arguments:

-h,  --help : Show this help message and exit

-i DATASET,  --dataset: The tubulated pharmaco-omic dataset (columns are Patient, their corresponding drug response, annotated as Responder or Non Responder, and molecular features)

-c CLASSIFIER, --classifier:  Machine learning algorithm employed to build classifiers, Possibilities are: - CART or - PCART (CART with OMC)

-p HYPERPARAM, --hyperparameters: Whether to apply hyperparameter tuning

-r NB_RANDOM_SEEDS, --random-seeds: Number of random seeds used to run cross-validation tests, Default = 5

-m METRIC, --metric:  Metric used to select optimized models, Default = MCC, Possibilities are: AUC, MCC, AUCPR, AVGPRE, PRE, REC, F1

-sw, --sample-weighting:  Whether to apply sample weighting

-log2, --log2-transformation: Whether to apply log2-transformation on real-value features in dataset

-imp, --imputation: Whether to apply imputation on missing values during model training

-t, --threshold:  Whether to adjust threshold for assigning sample class according to its class probability

-n, --normalize:  Whether to normalize the dataset (apply standard normalization prior to training)

Examples of command lines:
- Building the CART model using the isomiR dataset (provided above) by 10-fold cross-validation 5 repetition, apply sample weight and normalize the data: 
$ CART_model.py -i isomiR.tsv -c CART -r 5 -m MCC -sw -n

 
