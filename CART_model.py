#!/usr/bin/python
#-*- coding:utf-8 -*-

"""
This program performs nested 10 stratified Cross-validation (10CV) to evaluate optimized
predictive models based on datasets derived from the TCGA (https://gdc.cancer.gov/).

Tuning is done for:
	- Machine learning algorithm hyperparameters
	- Optimal Model Complexity (OMC)
	- Operating threshold

A deployment phase (i.e. optimized models trained on all the input dataset) is done to
	build a model that can be employed directly on new/independent data.

Standard CVs are used for comparing the optimized models to:
	- Default classifier (i.e. default hyperparameters, no feature selection
		and default operating threshold)
	- Permutated model (i.e. all-features model with sample labels permutated
		during model training)

Input: tabulated file containing tumour identifiers within the column "Patient",
	their corresponding drug response within the column "response", annotated as "Responder"
	or "NonResponder", and molecular profile (the rest of the columns contain features).

Outputs:
	- Analysis performed during model selection
                ("inner_loop_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv")

	- Analysis performed during model evaluation
                ("model_assessment_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv")

	- Analysis performed during deployment phase
		("deployment_phase_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv")

	- Classification scores obtained from optimized
                ("confusion_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv"), default
                ("default_[ALGORITHM]_predictive_performance.tsv") and permutated
                ("permutation_test_[ALGORITHM]_predictive_performance.tsv") models

	- Confusion matrices obtained from predictions of optimized models
                ("confusion_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv") and default
                ("default_confusion_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv")
		

	- List of selected features (if OMC investigated) from inner loop
                ("common_selected_features_from_inner_loops_seed[SEED].tsv") and deployment phase
                ("selected_features_from_deployment_phase_seed[SEED].tsv")

	- Classifiers obtained from the deployment phase
                ("model_[ALGORITHM]_[PARAM(S)]_seed[SEED].pkl")


	- Coordinates to draw ROC curves obtained from the predictions returned by optimized and
                default models ("outer_test_folds_thresholds_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED]
                .tsv" and "default_thresholds_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv",
                respectively)

	- Class probabilities obtained from models built during inner and outer loops,
                and default models ("inner_loop[ITERATION]_outer_train_folds_class_proba_matrix_
                [ALGORITHM]_[PARAMS]_seed[SEED].tsv",
                "outer_test_folds_class_proba_matrix_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv",
                "default_class_proba_[ALGORITHM]_[PARAM(S)]_seed[SEED].tsv")

	Outputs related to optimized models are found in the output directory
	"hyperparameter_tuning_[ALGORITHM]_[PARAM(S)]_[DATE]".

Execution: The program should be launched from the working directory.

Usage: CART.py [-h] -i DATASET -c CLASSIFIER [-p HYPERPARAM]
                          [-r NB_RANDOM_SEEDS] [-m METRIC] [-sw] [-log2]
                          [-imp] [-t]

Optional arguments:
  -h, --help            Show this help message and exit.
  -i DATASET, --data DATASET
                        Pharmaco-omic dataset.
  -c CLASSIFIER, --classifier CLASSIFIER
                        Machine learning algorithm employed to build
                        classifiers. Possibilities are:
                        - CART
                        - PCART (CART with OMC)
  -p HYPERPARAM, --hyperparameters HYPERPARAM
                        Whether to apply hyperparameter tuning
  -r NB_RANDOM_SEEDS, --nb-random-seeds NB_RANDOM_SEEDS
                        Number of random seeds used to run cross-validation.
                        tests. Default = 5
  -m METRIC, --metric METRIC
                        Metric used to select optimized models. Default = MCC
                        Possibilities are: AUC, MCC, AUCPR, AVGPRE, PRE, REC, F1
  -sw, --sample-weighting
                        Whether to apply sample weighting.
  -log2, --log2-transformation
                        Whether to apply log2-transformation on real-value
                        features in dataset.
  -imp, --imputation    Whether to apply imputation on missing values during
                        model training.
  -t, --threshold       Whether to adjust threshold for assigning sample class
                        according to its class probability.
  -n, --normalize       Whether to normalize the dataset
  						(apply standard normaliztion prior to training).

Examples of command lines:
	- Tuning of model complexity only:
		CART_model.py
			--input  InputDataset.tsv
			--classifier  PCART
			--nb-random-seeds  10
			--metric  MCC
			--threshold
			--sample-weighting

	- Tuning of some hyperparameters simultaneously:
		CART_model.py
			--input InputDataset.tsv
			--classifier CART
			--hyperparameters 
			--nb-random-seeds 10
			--metric MCC
			--sample-weighting

Dependencies:
	- pandas (1.2.3)
	- numpy (1.15.4)
	- scikit-learn (0.20.1)
	- imbalanced-learn (0.3.3)
	- dtreeviz (1.2)
	- joblib (1.0.1)
	- graphviz (0.16)
	- matplotlib (3.3.4)
	- numpy (1.20.2)
	- scikit-learn (0.24.1)
	- seaborn (0.11.1)
"""
import pandas as pd
from pandas import DataFrame
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
from textwrap import wrap
from IPython.display import Image
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.tree import export_graphviz
from dtreeviz.trees import *
from IPython.display import Image, display_svg, SVG
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold, KFold, permutation_test_score, GridSearchCV, cross_val_score, cross_val_predict, ParameterGrid
from sklearn.metrics import matthews_corrcoef, confusion_matrix, roc_curve, precision_recall_curve, auc, make_scorer, roc_auc_score, accuracy_score, average_precision_score, recall_score, precision_score, f1_score
from imblearn.pipeline import Pipeline,  make_pipeline
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.base import clone
from itertools import cycle
import sys, argparse, os, json, time, re, copy, glob, threading, multiprocessing, joblib, itertools, pydotplus
from scipy import interp
from colorsys import hls_to_rgb
from pprint import pprint
from joblib import Parallel, delayed


## For reproducibility
np.random.seed(0)

########################################################################
def mcc_scorer(actual, prediction):
	"""This function calculates Matthews Correlation Coefficient (MCC).

	@type	actual: array-like
	@param	actual: Actual sample labels.
	@type	prediction: array-like
	@param	prediction: Predicted sample labels.

	@return: Calculated MCC from predictions. Returns NaN if a class is not predicted at all.
	"""
	with np.errstate(all = 'raise'):
			try:
				return matthews_corrcoef(actual, prediction)
			except FloatingPointError:
				return np.nan

########################################################################
def drawdtreeviz(file_name, clf, X, y, target_name, feature_names, l_class_names):
	"""This function draw the decision tree in treeviz form
	@type	clf: machine learning algorithm object
	@param	clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: The data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type   target_name: list
	@param  target_name:  list of target names
	@type   feature_names: list
	@param	feature_names: list of features names
	@type   l_class_names: list 
	@param	l_class_names: list of class_names

	@return: decision tree in dtreeviz.
	"""
	try:
		viz = dtreeviz(clf, X, y, target_name = target_name, feature_names = feature_names, class_names = l_class_names)
	except AttributeError:
		viz = dtreeviz(clf.named_steps['model'], X, y, target_name = target_name, feature_names = feature_names, class_names = l_class_names)
	viz.save(file_name)

#######################################################################

def cross_val_predictProba_10cv(clf, X, y, sample_weight, cv):
	"""This function performs standard 10CV.

	@type	clf: machine learning algorithm object
	@param	clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: The data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	cv: cross-validation generator, LeaveOneOut object
	@param	cv: 10CV generator.

	@return: Predictions and class probabilities.
	"""
	np.random.seed(0)
	df_pred = pd.DataFrame()
	y_pred = np.array([])
	df_proba = pd.DataFrame()
	y_proba = np.empty((0, 2))
	cv_splits = cv.split(X,y)

	o = 1

	for train_index, test_index in cv_splits:
		Xtrain = X[train_index]
		Xtest = X[test_index]

		ytrain = y[train_index]
		ytest = y[test_index]

		cclf = clone(clf)

		if sample_weight:
			if not isinstance(cclf, Pipeline):
				cclf.fit(Xtrain, ytrain, sample_weight = compute_sample_weight("balanced", ytrain))
			else:
				cclf.fit(Xtrain, ytrain, model__sample_weight = compute_sample_weight("balanced", ytrain))
		else:
			cclf.fit(Xtrain, ytrain)

		if df_proba.empty:
			df_proba = pd.DataFrame(cclf.predict_proba(Xtest))
			df_proba['ind'] = test_index
		else:
			current_proba = pd.DataFrame(cclf.predict_proba(Xtest))
			current_proba['ind'] = test_index
			df_proba = pd.concat([df_proba, current_proba])

		if df_pred.empty:
			df_pred = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
		else:
			current_pred = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
			df_pred = pd.concat([df_pred, current_pred])

		print(str(o) + " predictions out of " + str(X.shape[0]))
		o = o + 1

	#print(df_proba)
	df_proba = df_proba.sort_values('ind')
	#print(df_proba)
	y_proba = df_proba.drop(['ind'], axis = 1).values

	#print(df_pred)
	df_pred = df_pred.sort_values('ind')
	#print(df_pred)
	y_pred = df_pred['pred'].values

	return y_pred, y_proba, cclf

########################################################################

def select_adjusted_threshold(y, y_proba):
	"""This function selects optimal operating threshold based on class probabilites.

	@type	y: array-like
	@param	y: Actual sample labels.
	@type	y_proba: array-like
	@param	y_proba: Calculated class probabilities.

	@return: The selected operating threshold, and the classifications scores obtained employing this threshold.
	"""
	thresholds = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
	l_tuples_thr_mcc = []

	for t in thresholds:
		y_pred_thr = np.where(y_proba[:,1] > t, 1, 0)
		mcc_thr = mcc_scorer(y, y_pred_thr)
		l_tuples_thr_mcc.append((t, mcc_thr, y_pred_thr))

	try:
		tup_thr_with_highest_mcc = sorted([tup for tup in l_tuples_thr_mcc if tup[1] == np.nanmax([x[1] for x in l_tuples_thr_mcc])], key = lambda x:x[0])[0]

	except IndexError:
		for tup in l_tuples_thr_mcc:
			if tup[0] == 0.5:
				tup_thr_with_highest_mcc = tup

	thr_with_highest_mcc = tup_thr_with_highest_mcc[0]
	highest_mcc = tup_thr_with_highest_mcc[1]
	pre_thr_highest_mcc = precision_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
	rec_thr_highest_mcc = recall_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
	f1_thr_highest_mcc = f1_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
	accu_thr_highest_mcc = accuracy_score(y, tup_thr_with_highest_mcc[2])

	return thr_with_highest_mcc, highest_mcc, pre_thr_highest_mcc, rec_thr_highest_mcc, f1_thr_highest_mcc, accu_thr_highest_mcc

########################################################################

def cross_val_predictProba_10cv_paramsEvaluation(clf, p_grid, X, y, sample_weight, cv, l_mean_prediction_time, l_mcc, l_auc, l_avgpre, l_pre, l_rec, l_f1, l_accu, l_aucpr, threshold, l_thr = None):
	"""This function performs model selection via inner standard 10CV.

	@type	clf: machine learning algorithm object
	@param	clf: The machine learning algorithm to use to fit the data.
	@type	p_grid: dictionary
	@param	p_grid: Tuned hyperparameters.
	@type	X: array-like
	@param	X: Data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	cv: cross-validation generator, LeaveOneOut object
	@param	cv: 10CV generator for inner loop.
	@type	l_mean_prediction_time: list
	@param:	l_mean_prediction_time: Empty list to store mean prediction times.
	@type	l_accu: list
	@param:	l_accu: Empty list to store calculated accuracy score.
	@type	l_mcc: list
	@param:	l_mcc: Empty list to store calculated MCCs.
	@type	l_auc: list
	@param:	l_auc: Empty list to store calculated AUCs.
	@type	l_avgpre: list
	@param:	l_avgpre: Empty list to store calculated Average Precisions (AVGPREs).
	@type	l_pre: list
	@param:	l_pre: Empty list to store calculated PREs.
	@type	l_rec: list
	@param:	l_rec: Empty list to store calculated RECs.
	@type	l_f1: list
	@param:	l_f1: Empty list to store calculated F1s.
	@type	threshold:bool
	@param:	threshold: Whether to tune operating threshold.
	@type	l_thr: list
	@param:	l_thr: Empty list to store calculated optimized operating threshold.

	@return: None
	"""
	y_pred_proba_params = {}
	cv_splits = cv.split(X,y)

	if threshold:
		if l_thr is None:
			print("error : you have to provide a list of thresholds to fill")

	o = 1

	for train_index, test_index in cv_splits:
		for p in ParameterGrid(p_grid):
			print(time.strftime("%Y-%m-%d at %H:%M:%S"))
			print("Sample " + str(o)  + " being processed with " + str(p))

			start_predict = time.time()

			Xtrain = X[train_index]
			ytrain = y[train_index]

			Xtest = X[test_index]
			ytest = y[test_index]

			print("INNER FOLD: y_train, y_test")
			print(str(o) + ": " + str(y[train_index]) + ", " + str(y[test_index]))
			
			tuned_clf = clone(clf).set_params(**p)

			#print("TUNED CLF")
			#print(tuned_clf.steps)

			if sample_weight:
				if not isinstance(tuned_clf, Pipeline):
					tuned_clf.fit(Xtrain, ytrain, sample_weight = compute_sample_weight("balanced", ytrain))
				else:
					tuned_clf.fit(Xtrain, ytrain, model__sample_weight = compute_sample_weight("balanced", ytrain))
			else:
				tuned_clf.fit(Xtrain, ytrain)

			df_proba_p = pd.DataFrame(tuned_clf.predict_proba(Xtest))
			df_proba_p['ind'] = test_index

			if not threshold:
				df_pred_p = pd.DataFrame({'ind':test_index, 'pred':tuned_clf.predict(Xtest)})

			end_predict = time.time()
			prediction_time = end_predict - start_predict

			try:
				y_pred_proba_params[str(p)] = list(y_pred_proba_params[str(p)])

				if not threshold:
					y_pred_proba_params[str(p)][0] = pd.concat([y_pred_proba_params[str(p)][0], df_pred_p])
				else:
					y_pred_proba_params[str(p)][0] = []

				y_pred_proba_params[str(p)][1] = pd.concat([y_pred_proba_params[str(p)][1], df_proba_p])
				y_pred_proba_params[str(p)][2] += prediction_time
				y_pred_proba_params[str(p)] = tuple(y_pred_proba_params[str(p)])

			except KeyError:
				if not threshold:
					y_pred_proba_params[str(p)] = (df_pred_p, df_proba_p, prediction_time)
				else:
					y_pred_proba_params[str(p)] = ([], df_proba_p, prediction_time)

				y_pred_proba_params[str(p)] = list(y_pred_proba_params[str(p)])
				y_pred_proba_params[str(p)] = tuple(y_pred_proba_params[str(p)])

			print(time.strftime("%Y-%m-%d at %H:%M:%S"))
			print(str(p) + 	" processed.")

		print(str(o)  + " predictions out of " + str(X.shape[0]))
		o = o + 1

	for p in ParameterGrid(p_grid):
		y_pred_proba_params[str(p)] = list(y_pred_proba_params[str(p)])
		try:
			y_pred_proba_params[str(p)][0] = y_pred_proba_params[str(p)][0].sort_values('ind') # predictions
		except AttributeError:
			pass

		print(y_pred_proba_params[str(p)][1])
		y_pred_proba_params[str(p)][1] = y_pred_proba_params[str(p)][1].sort_values('ind') # class probabilities
		print(y_pred_proba_params[str(p)][1])
		y_pred_proba_params[str(p)] = tuple(y_pred_proba_params[str(p)])

	for p in ParameterGrid(p_grid):
		print(y_pred_proba_params[str(p)][1])
		try:
			fpr, tpr, thresholds = roc_curve(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).values[:,1], drop_intermediate = False)
			precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).values[:,1])

		except ValueError:
			fpr, tpr, thresholds = roc_curve(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).fillna(0).values[:,1], drop_intermediate = False)
			precision, recall, thresholds = precision_recall_curve(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).fillna(0).values[:,1])

		if threshold:
			thr_p, mcc_p, pre_p, rec_p, f1_p, accu_p = select_adjusted_threshold(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).values)
			l_thr.append(thr_p)

		else:
			mcc_p = mcc_scorer(y, y_pred_proba_params[str(p)][0].drop(['ind'], axis = 1).values)
			pre_p = precision_score(y, y_pred_proba_params[str(p)][0].drop(['ind'], axis = 1).values, average = 'weighted')
			rec_p = recall_score(y, y_pred_proba_params[str(p)][0].drop(['ind'], axis = 1).values, average = 'weighted')
			f1_p = f1_score(y, y_pred_proba_params[str(p)][0].drop(['ind'], axis = 1).values, average = 'weighted')
			accu_p = accuracy_score(y, y_pred_proba_params[str(p)][0].drop(['ind'], axis = 1).values)

		auc_p = auc(fpr, tpr)
		aucpr_p = auc(recall,precision)
		avgpre_p = average_precision_score(y, y_pred_proba_params[str(p)][1].drop(['ind'], axis = 1).values[:,1], average = 'weighted')

		l_mcc.append(mcc_p)
		l_auc.append(auc_p)
		l_avgpre.append(avgpre_p)
		l_pre.append(pre_p)
		l_rec.append(rec_p)
		l_f1.append(f1_p)
		l_accu.append(accu_p)
		l_aucpr.append(aucpr_p)
		l_mean_prediction_time.append(y_pred_proba_params[str(p)][2]/len(y_pred_proba_params[str(p)][1]))

########################################################################

def nested_cross_validation(args):
	"""This function performs model selection via inner 10CV on outer training folds and predicts held-out samples. It writes class probabilities and coordinates of ROC curves obtained from the predictions of models resulting of the inner loop.

	@type	args: array
	@param	args: Array containing following input arguments for this function: index of iteration of 10CV, data from outer training fold, sample labels from outer training fold, data from outer test fold, sample labels from outer test fold, hyperparameter(s) to tune, machine learning algorithm employed, current random seed, whether to apply sample weighting, classification metric for model selection, list of features, whether to tune operating threshold.

	@return: index of iteration of 10CV, predictions from optimized models, selected hyperparameters, optimized operating thresholds, best classification score from inner loop, list of features selected from inner loop, machine learning algorithm employed, hyperparameter(s) to tune, output directory of model optimization.
	"""
	## For reproducibility with multi-threading
	np.random.seed(0) # useless with LeaveOneOut

	## 1. Split data into outer train and test folds
	i, outer_test_idx, X_outer_train, y_outer_train, X_outer_test, y_outer_test, p_grid, clf, seed, sample_weight, metric, l_feat, target_name, l_class_names, data, threshold = args

	# Metric considered to choose best model
	reference_metric = metric

	# Cross-validation modalities
	inner_cv = StratifiedKFold(n_splits = 10)

	# Basic classifiers
	if clf == 'CART':
		base_clf = DecisionTreeClassifier(random_state = seed)

		directory = "hyperparameter_tuning_CART_" + '_'.join(p_grid.keys()) + '_' + date

		try:
			if not os.path.exists(directory):
				os.makedirs(directory)

		except OSError:
			pass


	elif clf == 'PCART':
		if ~((X != 0) & (X != 1)).any():
			selectobject = SelectKBestPvalueFisherExact()
		else:
			selectobject = SelectKBest()

		base_clf = Pipeline([("selectfeat", selectobject), ("model", DecisionTreeClassifier(random_state = seed))])
		p_grid['selectfeat__k'] = range(2, int(X_outer_train.shape[0]/2) + 1)

		directory = "hyperparameter_tuning_FeatureSelection_CART_" + '_'.join([s.replace(s[0:s.find('__') + 2],'') for s in sorted(p_grid.keys())]) + '_' + date

		try:
			if not os.path.exists(directory):
				os.makedirs(directory)

		except OSError:
			pass

	
	# Nested 10CV
	## 1. Split data into outer train and test folds

	# Clone p_grid, if modif required (imputation step added : clf becomes a Pipeline)
	p_grid_tuning = copy.deepcopy(p_grid)

	# Initialisation dict of current inner loop
	dict_inner_loop = {"Random seed":[seed] * len(list(ParameterGrid(p_grid_tuning))), "Inner loop #":[i] * len(list(ParameterGrid(p_grid_tuning))), "params":list(ParameterGrid(p_grid_tuning))}

	# Check if missing values in X_outer_train. If yes --> first step of Pipeline = IMPUTATION
	if isinstance(base_clf, Pipeline): # in this case we perform feature selection --> imputation required
		if pd.DataFrame(X_outer_train).isnull().any().any():
			clf_to_tune = clone(base_clf)
			clf_to_tune.steps.insert(0, ("imputer", SimpleImputer()))
		else:
			clf_to_tune = clone(base_clf)

	elif isinstance(base_clf, DecisionTreeClassifier): # no feature selection --> the clf is converted into Pipeline to include imputation
		if pd.DataFrame(X_outer_train).isnull().any().any():
			clf_to_tune = Pipeline([("imputer", SimpleImputer()), ("model", clone(base_clf))])
			for k in p_grid_tuning.keys():
				p_grid_tuning['model__' + k] = p_grid_tuning.pop(k)
		else:
			clf_to_tune = clone(base_clf)

	else: # simple XGBoost & LightGBM
		clf_to_tune = clone(base_clf)

	# Initialization list scores, prediction times
	l_inner_mcc = []
	l_inner_auc = []
	l_inner_avgpre = []
	l_inner_pre = []
	l_inner_rec = []
	l_inner_f1 = []
	l_inner_accu = []
	l_inner_aucpr = []
	l_mean_prediction_time = []
	l_inner_selected_features = []

	if threshold:
		l_inner_thresholds = []
	else:
		l_inner_thresholds = None

	## 2. For each param : list preditions on outer train fold, then calculate inner AUC and MCC
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("10CV inner loop " + str(i))
	cross_val_predictProba_10cv_paramsEvaluation(clf_to_tune, p_grid_tuning, X_outer_train, y_outer_train, sample_weight, inner_cv, l_mean_prediction_time, l_inner_mcc, l_inner_auc, l_inner_avgpre, l_inner_pre, l_inner_rec, l_inner_f1, l_inner_accu, l_inner_aucpr, threshold, l_inner_thresholds)
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("End 10CV inner loop " + str(i))

	dict_inner_loop['mean_predictions_time (in seconds)'] = l_mean_prediction_time
	dict_inner_loop['inner_AUC'] = l_inner_auc
	dict_inner_loop['inner_AUCPR'] = l_inner_aucpr
	dict_inner_loop['inner_MCC'] = l_inner_mcc
	dict_inner_loop['inner_AVGPRE'] = l_inner_avgpre
	dict_inner_loop['inner_PRE'] = l_inner_pre
	dict_inner_loop['inner_REC'] = l_inner_rec
	dict_inner_loop['inner_F1'] = l_inner_f1
	dict_inner_loop['inner_ACCU'] = l_inner_accu

	if threshold:
		dict_inner_loop['inner_threshold'] = l_inner_thresholds

	## 3. Select best params, based on inner scores

	# add rank in inner loop dataframe to select best model
	df_inner_loop = pd.DataFrame(dict_inner_loop)
	df_inner_loop['inner_AUC'] = df_inner_loop['inner_AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_AUCPR'] = df_inner_loop['inner_AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_MCC'] = df_inner_loop['inner_MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_AVGPRE'] = df_inner_loop['inner_AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_PRE'] = df_inner_loop['inner_PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_REC'] = df_inner_loop['inner_REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_F1'] = df_inner_loop['inner_F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_inner_loop['inner_ACCU'] = df_inner_loop['inner_ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)

	df_inner_loop['rank_test_AUC'] = df_inner_loop['inner_AUC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_AUCPR'] = df_inner_loop['inner_AUCPR'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_MCC'] = df_inner_loop['inner_MCC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_AVGPRE'] = df_inner_loop['inner_AVGPRE'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_PRE'] = df_inner_loop['inner_PRE'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_REC'] = df_inner_loop['inner_REC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_F1'] = df_inner_loop['inner_F1'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_inner_loop['rank_test_ACCU'] = df_inner_loop['inner_ACCU'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)

	title_checkpoint_innerloop = directory + "/" + "checkpoint_inner_loop" + str(i) + "_" + str(clf_to_tune)[0:str(clf_to_tune).find('(')] + "_".join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	title_checkpoint_innerloop_open = open(title_checkpoint_innerloop, 'w')
	df_inner_loop.to_csv(title_checkpoint_innerloop_open, sep = '\t', index = False)
	title_checkpoint_innerloop_open.close()

	# select best estimator
	best_index = df_inner_loop["rank_test_%s" % reference_metric].argmin()
	best_params = df_inner_loop['params'][best_index]

	if threshold:
		best_threshold = df_inner_loop['inner_threshold'][best_index]
	else:
		best_threshold = None

	best_inner_score = df_inner_loop["inner_%s" % reference_metric][best_index]
	best_estimator = clone(clf_to_tune).set_params(**best_params)

	# write class probas of best model applied on inner train fold
	y_pred_best_on_inner_train_fold, y_proba_best_on_inner_train_fold, clf_inner_train_fold = cross_val_predictProba_10cv(best_estimator, X_outer_train, y_outer_train, sample_weight, inner_cv)
	df_proba_best_on_inner_train_fold = pd.DataFrame(y_proba_best_on_inner_train_fold)
	title_proba_best_on_inner_train_fold = directory + '/' + "inner_loop" + str(i) + "_outer_train_folds_class_proba_matrix_" + str(best_estimator)[0:str(best_estimator).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	open_proba_best_on_inner_train_fold = open(title_proba_best_on_inner_train_fold, 'w')
	df_proba_best_on_inner_train_fold.to_csv(open_proba_best_on_inner_train_fold, sep = '\t', index = False)
	open_proba_best_on_inner_train_fold.close()


	if sample_weight:
		if not isinstance(best_estimator, Pipeline):
			best_estimator.fit(X_outer_train, y_outer_train, sample_weight = compute_sample_weight("balanced", y_outer_train))
		else:
			best_estimator.fit(X_outer_train, y_outer_train, model__sample_weight = compute_sample_weight("balanced", y_outer_train))
	else:
		best_estimator.fit(X_outer_train, y_outer_train)

	# update set inner features if OMC. If not all features selected
	if isinstance(best_estimator, Pipeline):
		if 'selectfeat' in best_estimator.named_steps.keys():
			kept_best_feature_indices = best_estimator.named_steps['selectfeat'].get_support(indices = True)
			kept_best_feature_names = [l_feat[idx] for idx, _ in enumerate(l_feat) if idx in kept_best_feature_indices]
			if best_estimator.named_steps['selectfeat'].k != X.shape[1]:
				l_inner_selected_features = kept_best_feature_names
			else:
				l_inner_selected_features = []

	# get predictions and probabilities
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Prediction with best model in iteration " + str(i))

	y_proba_best = best_estimator.predict_proba(X_outer_test)

	if not threshold:
		y_pred_best = best_estimator.predict(X_outer_test)

	else:
		y_pred_best = np.where(y_proba_best[:,1] > best_threshold, 1, 0)

	print("Inner results " + str(i + 1) + " : ")
	print("Best estimator in inner loop " + str(i + 1) + " : ")
	print(best_estimator)
	print('\n')

	print("Best params in inner loop " + str(i + 1) + " : ")
	print(best_params)
	print('\n')

	print("Best " + reference_metric + ' in inner loop ' + str(i + 1) + " : ")
	print(best_inner_score)
	print('\n')

	return (i, outer_test_idx, y_pred_best, y_proba_best, best_params, best_threshold, best_inner_score, l_inner_selected_features, base_clf, p_grid, directory)

########################################################################

def calculate_random_seed_metrics(y, y_outer_test_preds, y_outer_test_probas, set_inner_selected_features, list_inner_best_params, list_inner_best_thresholds, list_inner_best_scores, base_clf, seed, directory, metric, p_grid):
	"""This function calculates classification metrics from predicted help-out samples (outer loop) and writes reports of analysis within inner loop.

	@type	y: array-like
	@param	y: Original sample labels.
	@type	y_outer_test_preds: array-like
	@param	y_outer_test_preds: Predicted sample labels of held-out samples (outer loop).
	@type	y_outer_test_probas: array-like
	@param	y_outer_test_probas: Class probabilities obtained for held-out samples (outer loop).
	@type	set_inner_selected_features: array-like
	@param	set_inner_selected_features: List of features commonly selected within inner loop.
	@type	list_inner_best_params: array-like
	@param	list_inner_best_params: List of selected hyperparameters.
	@type	list_inner_best_thresholds: array-like
	@param	list_inner_best_thresholds: List of optimized operating thresholds.
	@type	list_inner_best_scores: array-like
	@param	list_inner_best_scores: List of the best classification scores obtained from model selections.
	@type	base_clf: machine learning algorithm object
	@param	base_clf: The machine learning algorithm to use to fit the data.
	@type	seed: int
	@param	seed: Current random seed.
	@type	directory: string
	@param	directory: Output directory of analysis for model optimization.
	@type	metric: string
	@param	metric: Classification metric employed for model selection.
	@type	p_grid: dictionary
	@param	p_grid: Tuned hyperparameters.

	@return: Model assessment from outer loop (selected hyperparameters, predictions, classification scores).
	"""
	## 5. Calculate final metrics and write global inner loops reports

	# Metric considered to choose best model
	reference_metric = metric

	l_df_inner_loops = []

	fpr, tpr, auc_thresholds = roc_curve(y, y_outer_test_probas[:,1], drop_intermediate = False)
	final_auc = auc(fpr, tpr)
	precision, recall, aucpr_thresholds = precision_recall_curve(y, y_outer_test_probas[:,1])
	final_aucpr = auc(recall, precision)
	final_mcc = mcc_scorer(y, y_outer_test_preds)
	final_avgpre = average_precision_score(y, y_outer_test_probas[:,1], average = 'weighted')
	final_pre = precision_score(y, y_outer_test_preds, average = 'weighted')
	final_rec = recall_score(y, y_outer_test_preds, average = 'weighted')
	final_f1 = f1_score(y, y_outer_test_preds, average = 'weighted')
	final_accu = accuracy_score(y, y_outer_test_preds)

	# Concat check points inner loop here
	path_inner_checkpoints = directory + "/checkpoint_inner_loop*"
	l_path_inner_checkpoints = glob.glob(path_inner_checkpoints)

	for c in l_path_inner_checkpoints:
		df_i = pd.read_csv(c, sep ='\t', header = 0)
		l_df_inner_loops.append(df_i)

	df_inner_loops = pd.concat(l_df_inner_loops)
	df_inner_loops = df_inner_loops.reset_index(drop = True)

	# Write inner loops file for the current random_seed
	inner_title = directory + '/' + "inner_loop_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + '.tsv'
	inner_title_open = open(inner_title, 'w')
	df_inner_loops.to_csv(inner_title_open, sep = '\t', index = False)
	inner_title_open.close()

	# Remove check points
	for c in l_path_inner_checkpoints:
		os.remove(c)

	# If OMC : Write list of features selected during inner loops
	if isinstance(base_clf, Pipeline):
		if 'selectfeat' in base_clf.named_steps.keys():
			if set_inner_selected_features:
				title_inner_feat_list = directory + "/" + "common_selected_features_from_inner_loops_seed" + str(seed) + ".tsv"
				open_inner_feat_list = open(title_inner_feat_list, "w")
				for feat in set_inner_selected_features:
					open_inner_feat_list.write(feat + "\n")
				open_inner_feat_list.close()

			else:
				print("All features selected at each inner loop")

	## Write model assessment file
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Model assessment results : ")
	dict_tuned_param = {}
	for j in list_inner_best_params:
		for k,v in j.items():
			dict_tuned_param.setdefault(k, []).append(v)

	dict_outer_res = {'best_' + k:v for k,v in dict_tuned_param.items()}
	dict_outer_res['outer_train_' + reference_metric] = list_inner_best_scores

	outer_results = pd.DataFrame({k:dict_outer_res[k] for k in sorted(dict_outer_res.keys(), reverse = True)})
	outer_results = outer_results.reindex(outer_results.index.rename('outer_fold'))
	outer_results.index = range(1, len(outer_results) + 1)

	outer_title = directory + '/' + "model_assessment_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"

	if isinstance(base_clf, Pipeline):
		outer_results['params'] = '__'.join([s.replace(s[0:s.find('__') + 2],'') for s in sorted(p_grid.keys())])
	else:
		outer_results['params'] = '__'.join(sorted(p_grid.keys()))

	outer_results.reset_index(level = 0, inplace = True)
	outer_results = outer_results.rename(columns = {'index':'outer_fold'})

	outer_results['final_MCC'] = final_mcc
	outer_results['final_AUC'] = final_auc
	outer_results['final_AUCPR'] = final_aucpr
	outer_results['final_AVGPRE'] = final_avgpre
	outer_results['final_PRE'] = final_pre
	outer_results['final_REC'] = final_rec
	outer_results['final_F1'] = final_f1
	outer_results['final_ACCU'] = final_accu

	outer_results['final_MCC'] = outer_results['final_MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_AUC'] = outer_results['final_AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_AUCPR'] = outer_results['final_AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_AVGPRE'] = outer_results['final_AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_PRE'] = outer_results['final_PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_REC'] = outer_results['final_REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_F1'] = outer_results['final_F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['final_ACCU'] = outer_results['final_ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)
	outer_results['outer_train_' + reference_metric] = outer_results['outer_train_' + reference_metric].apply(lambda x: round(x, 3) if type(x) is float else x)

	outer_results['random_seed'] = seed

	if list_inner_best_thresholds:
		outer_results['threshold'] = list_inner_best_thresholds

	outer_results.to_csv(outer_title, sep = "\t", index_label = "outer_fold", index = False)

	predomc_title = directory + '/' + "prediction_omc_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	df_predomc = pd.DataFrame({'actual_label':y, 'predicted_label':y_outer_test_preds})
	df_predomc.to_csv(predomc_title, sep = '\t', index = False)

	# Confusion matrix
	title_confusion_matrix = directory + '/' + "confusion_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	open_confusion_matrix = open(title_confusion_matrix, 'w')
	tn, fp, fn, tp = confusion_matrix(y, y_outer_test_preds).ravel()
	df_confusion_matrix = pd.DataFrame.from_records([{"random_seed":seed, "tn":tn, "fp":fp, "fn":fn, "tp":tp, "MCC":final_mcc, "AUC":final_auc, "AUCPR":final_aucpr, "AVGPRE":final_avgpre, "PRE":final_pre, "REC":final_rec, "F1":final_f1, "ACCU":final_accu}])
	df_confusion_matrix['MCC'] = df_confusion_matrix['MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['AUC'] = df_confusion_matrix['AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['AUCPR'] = df_confusion_matrix['AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['AVGPRE'] = df_confusion_matrix['AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['PRE'] = df_confusion_matrix['PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['REC'] = df_confusion_matrix['REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['F1'] = df_confusion_matrix['F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix['ACCU'] = df_confusion_matrix['ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_confusion_matrix.to_csv(open_confusion_matrix, sep = '\t', index = False)
	open_confusion_matrix.close()

	# Draw ROC curves
	auc_thresholds_title = directory + '/' + "outer_test_folds_rocauc_plotting_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	auc_thresholds_open = open(auc_thresholds_title, 'w')
	auc_thresholds_df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
	auc_thresholds_df.to_csv(auc_thresholds_open, index = False, sep = "\t")
	auc_thresholds_open.close()

	auc_thresholds_title = directory + '/' + "outer_test_folds_rocauc_thresholds_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	auc_thresholds_open = open(auc_thresholds_title, 'w')
	auc_thresholds_df = pd.DataFrame(dict(thresholds = auc_thresholds))
	auc_thresholds_df.to_csv(auc_thresholds_open, index = False, sep = "\t")
	auc_thresholds_open.close()

	aucpr_thresholds_title = directory + '/' + "outer_test_folds_aucpr_plotting_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	aucpr_thresholds_open = open(aucpr_thresholds_title, 'w')
	aucpr_thresholds_df = pd.DataFrame(dict(recall = recall, precision = precision))
	aucpr_thresholds_df.to_csv(aucpr_thresholds_open, index = False, sep = "\t")
	aucpr_thresholds_open.close()

	aucpr_thresholds_title = directory + '/' + "outer_test_folds_aucpr_thresholds_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	aucpr_thresholds_open = open(aucpr_thresholds_title, 'w')
	aucpr_thresholds_df = pd.DataFrame(dict(thresholds = aucpr_thresholds))
	aucpr_thresholds_df.to_csv(aucpr_thresholds_open, index = False, sep = "\t")
	aucpr_thresholds_open.close()

	# Store class probabilities
	title_class_proba_matrix = directory + '/' + "outer_test_folds_class_proba_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	open_class_proba_matrix = open(title_class_proba_matrix, 'w')
	df_class_proba = pd.DataFrame(y_outer_test_probas)
	df_class_proba.to_csv(open_class_proba_matrix, sep = '\t', index = False)
	open_class_proba_matrix.close()

	#print(outer_results)
	return outer_results

########################################################################

def deployment_phase(p_grid, sample_weight, seed, base_clf, X, y, inner_cv, directory, metric, target_name, l_feat, l_class_names, data, threshold):
	"""
	This function performs deployment phase (i.e. training an optimized model on the entire dataset)

	@type	p_grid: dictionary
	@param	p_grid: Hyperparameters to tune.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	seed: int
	@param	seed: Current random seed.
	@type	base_clf: machine learning algorithm object
	@param	base_clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: Data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	inner_cv: cross-validation generator, LeaveOneOut object
	@param	inner_cv: 10CV generator.
	@type	directory: string
	@param	directory: Output directory of analysis for model optimization.
	@type	metric: string
	@param	metric: Classification metric employed for model selection.
	@type	l_feat: array-like
	@param	l_feat: List of features.
	@type	threshold: bool
	@param	threshold: Whether to tune operating threshold.

	@return: None
	"""
	## 6. Deployment phase

	# Metric considered to choose best model
	reference_metric = metric

	# Clone p_grid, if modif required (imputation step added : clf becomes a Pipeline)
	p_grid_depl = copy.deepcopy(p_grid)

	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Deployment phase")
	dict_depl = {"Random seed":[seed] * len(list(ParameterGrid(p_grid_depl))), "params":list(ParameterGrid(p_grid_depl))}
	l_depl_mcc = []
	l_depl_auc = []
	l_depl_aucpr = []
	l_depl_avgpre = []
	l_depl_pre = []
	l_depl_rec = []
	l_depl_f1 = []
	l_depl_accu = []
	l_prediction_time_depl = []

	if threshold:
		l_depl_thr = []
	else:
		l_depl_thr = None

	# Check if missing values in X. If yes --> first step = IMPUTATION
	if isinstance(base_clf, Pipeline): # in this case we perform feature selection
		if pd.DataFrame(X).isnull().any().any():
			clf_depl = clone(base_clf)
			clf_depl.steps.insert(0, ("imputer", SimpleImputer()))
		else:
			clf_depl = clone(base_clf)

		clf_depl.named_steps['model'].n_jobs = multiprocessing.cpu_count() - 2

	elif isinstance(base_clf, DecisionTreeClassifier):
		if pd.DataFrame(X).isnull().any().any():
			clf_depl = Pipeline([("imputer", SimpleImputer()), ("model", clone(base_clf))])
			clf_depl.named_steps['model'].n_jobs = multiprocessing.cpu_count() - 2
			for k in p_grid_depl.keys():
				p_grid_depl['model__' + k] = p_grid_depl.pop(k)
		else:
			clf_depl = clone(base_clf)
			clf_depl.n_jobs = multiprocessing.cpu_count() - 2
	else:
		clf_depl = clone(base_clf)
		clf_depl.n_jobs = multiprocessing.cpu_count() - 2

	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Deployment phase for random seed " + str(seed) + "...")
	cross_val_predictProba_10cv_paramsEvaluation(clf_depl, p_grid_depl, X, y, sample_weight, inner_cv, l_prediction_time_depl, l_depl_mcc, l_depl_auc, l_depl_avgpre, l_depl_pre, l_depl_rec, l_depl_f1, l_depl_accu, l_depl_aucpr, threshold, l_depl_thr)
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("End of deployment phase for random seed " + str(seed))

	dict_depl['mean_predictions_time (in seconds)'] = l_prediction_time_depl

	if threshold:
		dict_depl['threshold'] = l_depl_thr

	dict_depl['AUC'] = l_depl_auc
	dict_depl['AUCPR'] = l_depl_aucpr
	dict_depl['MCC'] = l_depl_mcc
	dict_depl['AVGPRE'] = l_depl_avgpre
	dict_depl['PRE'] = l_depl_pre
	dict_depl['REC'] = l_depl_rec
	dict_depl['F1'] = l_depl_f1
	dict_depl['ACCU'] = l_depl_accu

	df_depl = pd.DataFrame(dict_depl)
	df_depl['MCC'] = df_depl['MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['AUC'] = df_depl['AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['AUCPR'] = df_depl['AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['AVGPRE'] = df_depl['AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['PRE'] = df_depl['PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['REC'] = df_depl['REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['F1'] = df_depl['F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_depl['ACCU'] = df_depl['ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)

	df_depl['rank_test_AUC'] = df_depl['AUC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_AUCPR'] = df_depl['AUCPR'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_MCC'] = df_depl['MCC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_AVGPRE'] = df_depl['AVGPRE'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_PRE'] = df_depl['PRE'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_REC'] = df_depl['REC'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_F1'] = df_depl['F1'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)
	df_depl['rank_test_ACCU'] = df_depl['ACCU'].rank(ascending = False, method = 'min', na_option = 'bottom').astype(int)

	best_index_depl = df_depl["rank_test_%s" % reference_metric].argmin()
	best_params_depl = df_depl['params'][best_index_depl]
	best_estimator_depl = clone(clf_depl).set_params(**best_params_depl)

	# the all dataset is used as train set --> novel data will be used as test set
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Best estimator in deployment phase fitting for random seed " + str(seed) + "...")
	if sample_weight:
		if not isinstance(best_estimator_depl, Pipeline):
			best_estimator_depl.fit(X, y, sample_weight = compute_sample_weight("balanced", y))
		else:
			best_estimator_depl.fit(X, y, model__sample_weight = compute_sample_weight("balanced", y))
	else:
		best_estimator_depl.fit(X, y)

	depl_title = directory + '/' + "deployment_phase_" + str(best_estimator_depl)[0:str(best_estimator_depl).find('(')] + '_' + '_'.join(p_grid_depl.keys()) + "_seed" + str(seed) + ".tsv"
	df_depl.to_csv(depl_title, sep = "\t", index = False)

	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Save model from deployment phase")
	model_filename = directory + '/' + "model_" + str(best_estimator_depl)[0:str(best_estimator_depl).find('(')] + '_' + '_'.join(p_grid_depl.keys()) + "_seed" + str(seed) + ".pkl"
	joblib.dump(best_estimator_depl, model_filename)

	# If OMC : store features kept in the selected model during deployment phase
	if isinstance(best_estimator_depl, Pipeline):
		if 'selectfeat' in best_estimator_depl.named_steps.keys():
			kept_best_feature_indices_depl = best_estimator_depl.named_steps['selectfeat'].get_support(indices = True)
			kept_best_feature_names_depl = [l_feat[idx] for idx, _ in enumerate(l_feat) if idx in kept_best_feature_indices_depl]

		title_deploymentphase_feat_list = directory + "/" + "selected_features_from_deployment_phase_seed" + str(seed) + ".tsv"
		open_deploymentphase_feat_list = open(title_deploymentphase_feat_list, "w")
		for feat in kept_best_feature_names_depl:
			open_deploymentphase_feat_list.write(feat + "\n")
		open_deploymentphase_feat_list.close()
#############################################################################################################
## Decision Tree in dtreeviz
	if  clf.upper() == 'PCART':
		dt_name = "OMC_features" + "seed" + str(seed) + '_' + date
		tree_filename = directory + '/' + dt_name + "_tree_PCART" + '.svg'
		data_omc =  data.loc[:,kept_best_feature_names_depl]
		print(data_omc.head(2))
		x_omc = data_omc.values
		print(x_omc)
		dt = DecisionTreeClassifier(random_state = seed)
		dt.fit(x_omc,y)
		drawdtreeviz(tree_filename, dt, x_omc, y, target_name = target_name, feature_names = kept_best_feature_names_depl, l_class_names = l_class_names)
	else: pass

	########################################################################
def default_model_performance(base_clf, X, y, sample_weight, outer_cv, seed, p_grid, target_name, feature_names, l_class_names):
	"""
	This function uses standard 10CV to evaluate the predictive performance of default models (i.e. all-features considered during model training/no prior feature selection, default hyperparameters and default operating threshold).
	It writes the classification scores, confusion matrices and coordinates of ROC curves obtained from predictions of default models.

	@type	base_clf: machine learning algorithm object
	@param	base_clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: Data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	outer_cv: cross-validation generator, LeaveOneOut object
	@param	outer_cv: 10CV generator.
	@type	seed: int
	@param	seed: Current random seed.
	@type	p_grid: dictionary
	@param	p_grid: Hyperparameters to tune.

	@return: Dataframe containing classification scores reached by default models.
	"""
	np.random.seed(0)
	## 7. Performance of default clf (no feature selection, no tuning)
	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Performance all-features model")
	if isinstance(base_clf, Pipeline):
		default_clf = base_clf.named_steps['model']
	else:
		default_clf = clone(base_clf)

	if isinstance(default_clf, DecisionTreeClassifier):
		if pd.DataFrame(X).isnull().any().any():
			default_clf = Pipeline([("imputer", SimpleImputer()), ("model", default_clf)])
			default_clf.named_steps['model'].n_jobs = multiprocessing.cpu_count() - 2
		else:
			default_clf.n_jobs = multiprocessing.cpu_count() - 2

	print("10CV Cross-validation with default model for seed " + str(seed) + "...")
	y_pred_default, y_proba_default, default_model = cross_val_predictProba_10cv(default_clf, X, y, sample_weight, outer_cv)

###########DTreeviz###############################################
###Decision tree on training folds for 10 CV
	if clf.upper() == 'CART' or clf.upper() == 'PCART':
		dt_name = "all_features_CART_" + "seed" + str(seed) + '_' + date
		tree_filename = dt_name + "tree_CART" + '.svg'
		drawdtreeviz(tree_filename, default_model, X, y, target_name = target_name, feature_names = feature_names, l_class_names = l_class_names)
	else: pass

#################################################################

	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("End of all-features model performance assessment")

	fpr, tpr, auc_thresholds = roc_curve(y, y_proba_default[:,1], drop_intermediate = False)
	auc_default = auc(fpr, tpr)
	precision, recall, aucpr_thresholds = precision_recall_curve(y, y_proba_default[:,1])
	aucpr_default = auc(recall,precision)
	mcc_default = mcc_scorer(y, y_pred_default)
	avgpre_default = average_precision_score(y, y_proba_default[:,1], average = 'weighted')
	pre_default = precision_score(y, y_pred_default, average = 'weighted')
	rec_default = recall_score(y, y_pred_default, average = 'weighted')
	f1_default = f1_score(y, y_pred_default, average = 'weighted')
	accu_default = accuracy_score(y, y_pred_default)

	dict_default = {}
	dict_default['random_seed'] = seed
	dict_default['params'] = 'All-features'
	dict_default['MCC'] = mcc_default
	dict_default['AUC'] = auc_default
	dict_default['AUCPR'] = aucpr_default
	dict_default['AVGPRE'] = avgpre_default
	dict_default['PRE'] = pre_default
	dict_default['REC'] = rec_default
	dict_default['F1'] = f1_default
	dict_default['ACCU'] = accu_default

	df_default = pd.DataFrame.from_records([dict_default])

	df_default['MCC'] = df_default['MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['AUC'] = df_default['AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['AUCPR'] = df_default['AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['AVGPRE'] = df_default['AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['PRE'] = df_default['PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['REC'] = df_default['REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['F1'] = df_default['F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default['ACCU'] = df_default['ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)

	#print(df_default)

	title_checkpoint_default = "checkpoint_default_" + str(base_clf)[0:str(base_clf).find('(')] + "_seed" + str(seed) + ".tsv"
	title_checkpoint_default_open = open(title_checkpoint_default, 'w')
	df_default.to_csv(title_checkpoint_default_open, sep = '\t', index = False)
	title_checkpoint_default_open.close()

	# Default predictions
	default_preds_df = pd.DataFrame({'actual_label':y, 'predicted_label':y_pred_default})
	default_preds_df_title = "default_predictions_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	default_preds_df_open = open(default_preds_df_title, 'w')
	default_preds_df.to_csv(default_preds_df_open, sep = '\t', index = False)
	default_preds_df_open.close()

	# Confusion matrix
	title_default_confusion_matrix = "default_confusion_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	tn, fp, fn, tp = confusion_matrix(y, y_pred_default).ravel()
	df_default_confusion_matrix = pd.DataFrame.from_records([{"random_seed":seed, "tn":tn, "fp":fp, "fn":fn, "tp":tp, "MCC":mcc_default, "AUC":auc_default, "AUCPR":aucpr_default, "AVGPRE":avgpre_default, "PRE":pre_default, "REC":rec_default, "F1": f1_default, "ACCU": accu_default}])
	df_default_confusion_matrix['MCC'] = df_default_confusion_matrix['MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['AUC'] = df_default_confusion_matrix['AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['AUCPR'] = df_default_confusion_matrix['AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['AVGPRE'] = df_default_confusion_matrix['AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['PRE'] = df_default_confusion_matrix['PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['REC'] = df_default_confusion_matrix['REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['F1'] = df_default_confusion_matrix['F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_default_confusion_matrix['ACCU'] = df_default_confusion_matrix['ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)
	open_default_confusion_matrix = open(title_default_confusion_matrix, 'w')
	df_default_confusion_matrix.to_csv(open_default_confusion_matrix, sep = '\t', index = False)
	open_default_confusion_matrix.close()

	# Draw ROC curves
	default_auc_thresholds_title = "default_rocauc_plotting_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	dafault_auc_thresholds_df = pd.DataFrame(dict(fpr = fpr, tpr = tpr))
	default_auc_thresholds_open = open(default_auc_thresholds_title, 'w')
	dafault_auc_thresholds_df.to_csv(default_auc_thresholds_open, index = False, sep = "\t")
	default_auc_thresholds_open.close()

	default_auc_thresholds_title = "default_rocauc_thresholds_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	dafault_auc_thresholds_df = pd.DataFrame(dict(thresholds = auc_thresholds))
	default_auc_thresholds_open = open(default_auc_thresholds_title, 'w')
	dafault_auc_thresholds_df.to_csv(default_auc_thresholds_open, index = False, sep = "\t")
	default_auc_thresholds_open.close()

	default_aucpr_thresholds_title = "default_aucpr_plotting_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	dafault_aucpr_thresholds_df = pd.DataFrame(dict(recall = recall, precision = precision))
	default_aucpr_thresholds_open = open(default_aucpr_thresholds_title, 'w')
	dafault_aucpr_thresholds_df.to_csv(default_aucpr_thresholds_open, index = False, sep = "\t")
	default_aucpr_thresholds_open.close()

	default_aucpr_thresholds_title = "default_aucpr_thresholds_matrix_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	dafault_aucpr_thresholds_df = pd.DataFrame(dict(thresholds = aucpr_thresholds))
	default_aucpr_thresholds_open = open(default_aucpr_thresholds_title, 'w')
	dafault_aucpr_thresholds_df.to_csv(default_aucpr_thresholds_open, index = False, sep = "\t")
	default_aucpr_thresholds_open.close()

	# Store class probabilities
	default_class_proba_title = "default_class_proba_" + str(base_clf)[0:str(base_clf).find('(')] + '_' + '_'.join(p_grid.keys()) + "_seed" + str(seed) + ".tsv"
	default_class_proba_open = open(default_class_proba_title, 'w')
	default_class_proba_df = pd.DataFrame(y_proba_default)
	default_class_proba_df.to_csv(default_class_proba_open, sep = '\t', index = False)
	default_class_proba_open.close()

	return df_default

########################################################################

def permutation_test_10cv(base_clf, X, y, sample_weight, seed):
	"""This function performs permutation test on all-features model via standard 10CV.

	@type	base_clf: machine learning algorithm object
	@param	base_clf: The machine learning algorithm to use to fit the data.
	@type	X: array-like
	@param	X: The data to fit.
	@type	y: array-like
	@param	y: Original sample labels.
	@type	sample_weight: bool
	@param	sample_weight: Whether to apply sample weighting.
	@type	seed: int
	@param	seed: Random seed employed for the current iteration.

	@return: Dataframe containing classification scores reached by permutated models.
	"""
	## 8. Permutation test
	np.random.seed(0)

	print(time.strftime("%Y-%m-%d at %H:%M:%S"))
	print("Permutation test")
	if isinstance(base_clf, Pipeline):
		permut_clf = base_clf.named_steps['model']
	else:
		permut_clf = clone(base_clf)

	if isinstance(permut_clf, DecisionTreeClassifier):
		if pd.DataFrame(X).isnull().any().any():
			permut_clf = Pipeline([("imputer", SimpleImputer()), ("model", permut_clf)])
			permut_clf.named_steps['model'].n_jobs = multiprocessing.cpu_count() - 2
		else:
			permut_clf.n_jobs = multiprocessing.cpu_count() - 2

	df_preds = pd.DataFrame()
	df_probas = pd.DataFrame()
	cv_splits = StratifiedKFold(n_splits = 10).split(X, y)

	o = 1

	for train_index, test_index in cv_splits:
		Xtrain = X[train_index]
		Xtest = X[test_index]

		ytrain = y[train_index]
		ytest = y[test_index]

		ytrain_shuffled = np.random.permutation(ytrain)

		cclf = clone(permut_clf)

		if sample_weight:
			if not isinstance(cclf, Pipeline):
				cclf.fit(Xtrain, ytrain_shuffled, sample_weight = compute_sample_weight("balanced", ytrain_shuffled))
			else:
				cclf.fit(Xtrain, ytrain_shuffled, model__sample_weight = compute_sample_weight("balanced", ytrain_shuffled))
		else:
			cclf.fit(Xtrain, ytrain_shuffled)
		
		if df_preds.empty:
			df_preds = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
		else:
			current_df_preds = pd.DataFrame({'ind':test_index, 'pred':cclf.predict(Xtest)})
			df_preds = pd.concat([df_preds, current_df_preds])

		if df_probas.empty:
			df_probas = pd.DataFrame(cclf.predict_proba(Xtest))
			df_probas['ind'] = test_index
		else:
			current_df_probas = pd.DataFrame(cclf.predict_proba(Xtest))
			current_df_probas['ind'] = test_index
			df_probas = pd.concat([df_probas, current_df_probas])

		print(str(o)  + " permutated predictions out of " + str(X.shape[0]))
		o = o + 1

	df_preds = df_preds.sort_values('ind')
	y_pred = df_preds['pred'].values.astype(int)

	df_probas = df_probas.sort_values('ind')
	y_proba = df_probas.drop(['ind'], axis = 1).values

	permutation_mcc = mcc_scorer(y, y_pred)
	fpr, tpr, thresholds = roc_curve(y, y_proba[:,1], drop_intermediate = False)
	permutation_auc = auc(fpr, tpr)
	precision, recall, thresholds = precision_recall_curve(y, y_proba[:,1])
	permutation_aucpr = auc(recall,precision)
	permutation_pre = precision_score(y, y_pred, average = 'weighted')
	permutation_avgpre = average_precision_score(y, y_proba[:,1], average = 'weighted')
	permutation_rec = recall_score(y, y_pred, average = 'weighted')
	permutation_f1 = f1_score(y, y_pred, average = 'weighted')
	permutation_accu = accuracy_score(y, y_pred)

	df_permutation = pd.DataFrame.from_records([{'random_seed':seed,'params':'Permutated 10CV','MCC':permutation_mcc,'AUC':permutation_auc,'AUCPR':permutation_aucpr,'AVGPRE':permutation_avgpre,'PRE':permutation_pre,'REC':permutation_rec, 'F1':permutation_f1, 'ACCU':permutation_accu}])
	df_permutation['MCC'] = df_permutation['MCC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['AUC'] = df_permutation['AUC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['AUCPR'] = df_permutation['AUCPR'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['AVGPRE'] = df_permutation['AVGPRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['PRE'] = df_permutation['PRE'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['REC'] = df_permutation['REC'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['F1'] = df_permutation['F1'].apply(lambda x: round(x, 3) if type(x) is float else x)
	df_permutation['ACCU'] = df_permutation['ACCU'].apply(lambda x: round(x, 3) if type(x) is float else x)

	#print(df_permutation)

	title_checkpoint_permut = "checkpoint_permutation_test_" + str(base_clf)[0:str(base_clf).find('(')] + "_seed" + str(seed) + ".tsv"
	title_checkpoint_permut_open = open(title_checkpoint_permut, 'w')
	df_permutation.to_csv(title_checkpoint_permut_open, sep = '\t', index = False)
	title_checkpoint_permut_open.close()

	return df_permutation

########################################################################
def boxplot_or_swarmplot_metric(clf, df_tuning, df_default, df_permutation, plot_filename, metric, nb_random_seeds, n_samples):
	plt.figure()

	df_tuning.rename(columns = {'final_ACCU':'ACCU','final_GMEAN':'GMEAN','final_AUCPR':'AUCPR','final_AUC':'AUC','final_MCC':'MCC','final_AVGPRE':'AVGPRE','final_F1':'F1','final_PRE':'PRE','final_REC':'REC'}, inplace = True)
	df_tuning_sel = df_tuning.drop_duplicates(subset = ['AUC','MCC','random_seed','params'], keep = 'first')
	#df_tuning_sel.drop(['actual_label', 'outer_fold', 'predicted_label'], axis =1, inplace = True)
	df = pd.concat([df_tuning_sel, df_default,df_permutation])

	print(df)

	all_title = str(nb_random_seeds) + "_seeds_of_Nested_10CVs_" + str(n_samples) + "_data points_for_hyperparameters_tuning_of_" + clf + "_algorithm.tsv"
	df.to_csv(all_title, sep = '\t', index = False)

	## Boxplot
	ax = sns.boxplot(x = "params", y = metric, data = df, color = 'lightgrey')
	#ax = sns.swarmplot(x = 'params', y = metric, hue = 'random_seed', data = df, zorder = 1, color = 'black')
	#ax.legend_.remove()
	
	# Show number of observations on boxplots (https://python-graph-gallery.com/38-show-number-of-observation-on-boxplot/)
	# Calculate number of obs per group & max to position labels
	maxs = df.groupby(['params'])[metric].max().values
	maxs[np.isnan(maxs)] = 0
	#nobs = df['params'].value_counts().values --> error
	nobs =  df.groupby(['params'])[metric].count().values
	nobs = [str(x) for x in nobs.tolist()]
	nobs = ["n = " + i for i in nobs]
	# Add it to the plot
	pos = range(len(nobs))
	for tick, label in zip(pos, ax.get_xticklabels()):
		ax.text(pos[tick], maxs[tick] + 0.15, nobs[tick], horizontalalignment = 'center', size = 'x-small', color = 'black', weight = 'semibold')

	plt.xticks(rotation = 90)
	if metric == "MCC":
		plt.ylim(-1.05,1.05)
		plt.yticks(np.linspace(-1.0, 1.0, 11))
	elif "AUC" in metric:
		plt.ylim(-0.05,1.05)
		plt.yticks(np.linspace(0., 1.0, 11))
	elif "AUCPR" in metric:
		plt.ylim(-0.05,1.05)
		plt.yticks(np.linspace(0., 1.0, 11))
	
	if "AUC" in metric:
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.5], '--k', linewidth = 1)
	elif "AUCPR" in metric:
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.5], '--k', linewidth = 1)	
	elif metric == "MCC":
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.0], '--k', linewidth = 1)

	if "AUCPR" in metric:
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.5], '--k', linewidth = 1)
	elif "AUC" in metric:
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.5], '--k', linewidth = 1)	
	elif metric == "MCC":
		xlim = plt.xlim()
		plt.plot(xlim, 2 * [0.0], '--k', linewidth = 1)
		
	plt.ylabel(metric)
	plt.xlabel('Top k features vs All features (default)')
	
	t = metric + "s of " + str(nb_random_seeds) + " x Nested Leave-One-Out CVs (" + str(n_samples) + " data points) for hyperparameters tuning of " + clf + " algorithm"
		
	title = "\n".join(wrap(t, 60))
	plt.title(title, y = 1.08)
	plt.grid(True)
	plt.tight_layout()
	plt.savefig(plot_filename)

##################### Median of each seed #########################################
	med_MCC_omc= np.median(df_tuning_sel['MCC'])
	med_omc= df_tuning_sel.loc[df_tuning_sel['MCC'] == med_MCC_omc,:]

	med_MCC_default= np.median(df_default['MCC'])
	med_default= df_default.loc[df_default['MCC'] == med_MCC_default,:]

	med_MCC_permu= np.median(df_permutation['MCC'])
	med_permu= df_permutation.loc[df_permutation['MCC'] == med_MCC_permu,:]

	med_all = pd.concat([med_omc, med_default, med_permu])

	med_title = "median_" + str(nb_random_seeds) + "_seeds_of_Nested_10CVs_" + str(n_samples) + "_data points_for_hyperparameters_tuning_of_" + clf + "_algorithm.tsv"
	med_all.to_csv(med_title, sep = '\t', index = False)
	
	print(med_all)

########################################################################

if __name__=='__main__':
	## Declaration of arguments to argparse
	parser = argparse.ArgumentParser(add_help = True)
	parser.add_argument('-i','--input', action = 'store' , dest = 'dataset', required = True, help = 'Drug-cancer type molecular profile dataset')
	parser.add_argument('-c','--classifier', action = 'store', dest = 'classifier', default = 'CART', help = 'Classifier used')
	parser.add_argument('-p','--hyperparameters', action = 'store_true', dest = 'hyperparam', help = 'Whether to apply Hyperparameters to tune')
	parser.add_argument('-r','--nb-random-seeds', action = 'store', dest = 'nb_random_seeds', type = int, default = 5, help = 'Number of random seeds used to run cross-validation tests. Default = 10')
	parser.add_argument('-m','--metric', action = 'store', dest = 'metric', default = 'MCC', help = 'Metric used to evaluate the model performance')
	parser.add_argument('-sw','--sample-weighting', action = 'store_true', dest = 'sample_weight', help = 'Whether to aplly sample weighting')
	parser.add_argument('-log2','--log2-transformation', action = 'store_true', dest = 'log2', help = 'Whether to apply log2-transformation on real-value features in dataset')
	parser.add_argument('-imp','--imputation', action = 'store_true', dest = 'imputation', help = 'Whether to apply imputation on missing values during model training')
	parser.add_argument('-t','--threshold', dest = 'threshold', action = 'store_true', help = 'Whether to adjust threshold for assigning sample class according to its class probability.')
	parser.add_argument('-n','--normalize', action = 'store_true', dest = 'normalize', help = 'Whether to apply data normalization')
	arguments = parser.parse_args()

	inPut = arguments.dataset
	if not os.path.exists(inPut):
		print("No such file : " + inPut + ".")
		sys.exit(1)

	clf = arguments.classifier
	clf = clf.upper()

	if clf not in ['CART','PCART']:
		print("Classifier " + clf + " not supported.")
		print("Try :\n- CART : Classification And Regression Tree\n- PCART : CART preceded by feature selection")
		sys.exit(1)

	hparams = arguments.hyperparam

	if hparams:
		if (clf == 'CART') | (clf == 'PCART'):
			hparams = [{'model__min_samples_leaf': [1,2,3,4], 'model__max_features': [0.5,'auto','log2',None]}]
		else:
			pass
		
	else: # Only Optimal Model Complexity adjustment
		hparams = [{}]

	# NB : when we use feature selection, we have to add "model__" before the hyperparam name

	nb_random_seeds = arguments.nb_random_seeds

	metric = arguments.metric
	metric = metric.upper()
	if metric not in ['MCC','AUC','AVGPRE','PRE','REC','F1','AUCPR','ACCU']:
		print("Metric " + metric + " not supported.\nTry : MCC (Matthew' correlation coefficient) or\nAUC (Area Under the Curve) or\nAVGPRE (Average Precision) or\n PRE (Precision) or\n REC (Recall) or\n F1 (F1-score) or\n AUCPRE (Precision-Recall curve AUC)")
		sys.exit(1)

	sample_weight = arguments.sample_weight

	log2 = arguments.log2

	normalize = arguments.normalize

	imputation = arguments.imputation

	threshold = arguments.threshold

	# Log file initiation
	date = time.strftime("%Y-%m-%d")
	if clf == 'CART':
		tuned_algo = "Decision Tree"
		default_tsv = "default_CART_predictive_performance.tsv"
		permutation_tsv = "permutation_test_CART_predictive_performance.tsv"
		omc_tsv = "omc_CART_predictive_performance.tsv"

	
	elif clf == 'PCART':
		tuned_algo = "Decision Tree completed with OMC"
		default_tsv = "default_CART_predictive_performance.tsv"
		permutation_tsv = "permutation_test_CART_predictive_performance.tsv"
		omc_tsv = "omc_CART_predictive_performance.tsv"

	start_date = time.strftime("%Y-%m-%d at %H:%M:%S")

	# Read dataset to generate matrices
	data = pd.read_csv(inPut, sep = "\t", header = 0)
	data = data.dropna(axis = 1, how = 'any')

	if log2:
		data = pd.concat([data[['Patient ID_Aliquot ID', 'Drug response']], data.drop(['Patient ID_Aliquot ID', 'Drug response'], axis = 1).applymap(np.log2)], axis = 1)
		data = data.replace([np.inf, -np.inf], np.NaN)

	if not imputation:
		if data.isnull().any().any():
			data = data.dropna(axis = 1, how = 'any')

		# Data matrix
	data = data.replace("", np.nan)
	X = data.drop(['response','Patient'], axis = 1).values
	feature_names = data.drop(['response','Patient'], axis = 1).columns.tolist() 	# List of features
	
	#Use for dtreeviz only.
	target_name = data.drop(['Patient'], axis = 1).columns.tolist()[0]    # target_name
	class_names = np.array(list(set(data['response'].values))) # Class names
	l_class_names = list(class_names) #use by dtreeviz only(list of class_names)
	y = np.where(data['response'] == 'Responder', 1, 0) #Encoding labels

	##Normalize X if X conatins real numbers
	if normalize:
		if ~((X!=0) & (X!=1)).any():
			X = X
			print("X contains binary numbers as such it does not need normalization")
		else:
			scaler = StandardScaler()
			X = scaler.fit_transform(X)
			print("X contains real numbers as such it is normalized")

	# Random seed generator
	l_random_seeds = np.random.randint(0, 99999999, size = nb_random_seeds)

	global_results = []

	# MAIN = outer level + out of nested CV
	for seed in range(len(l_random_seeds)):
		for pgrid in hparams:
			y_outer_test_preds = np.array([])
			y_outer_test_probas = np.empty((0, 2))
			outer_test_indices = np.array([])
			list_inner_best_params = []
			list_inner_best_scores = []
			list_inner_best_thresholds = []

			if hparams == [{}]:
				set_inner_selected_features = set()
			else:
				set_inner_selected_features = None

			##outer_cv = LeaveOneOut()
			outer_cv = StratifiedKFold(n_splits = 10)
			outer_cv_splits = list(outer_cv.split(X,y))
			for t in enumerate(outer_cv_splits):
				print("FOLD: INDEX_test, y_train, y_test")
				print(str(t[0]) + ": " + str(t[1][1]) + ", " + str(y[t[1][0]]) + ", " + str(y[t[1][1]]))

			arg_instances = [(t[0], t[1][1], X[t[1][0]], y[t[1][0]], X[t[1][1]], y[t[1][1]], pgrid, clf, l_random_seeds[seed], sample_weight, metric, feature_names, target_name, l_class_names, data, threshold) for t in enumerate(outer_cv_splits)]
			#t[0]:fold, t[1][1]: test_index, X[t[1][0]:x_train,y[t[1][0]]:y_train, X[t[1][1]]:x_test, y[t[1][1]]:y_test
			nestedcv_results = Parallel(n_jobs = multiprocessing.cpu_count() - 2, verbose = 0, backend = "multiprocessing")(map(delayed(nested_cross_validation), arg_instances))

			base_clf_from_nestedcv_step = nestedcv_results[0][8]
			p_grid_from_nestedcv_step = nestedcv_results[0][9]
			directory_from_nestedcv_step = nestedcv_results[0][10]

			for s in nestedcv_results:
				print(s[1])
				print(s[2])
				print(s[3])
				outer_test_indices = np.append(outer_test_indices, s[1])
				y_outer_test_preds = np.append(y_outer_test_preds, s[2])
				y_outer_test_probas = np.append(y_outer_test_probas, s[3], axis = 0)

			try: # StratifiedKCV
				print(y_outer_test_preds)
				print(outer_test_indices)
				y_outer_test_preds = [o for l in y_outer_test_preds for o in l]
				outer_test_indices = [o for l in outer_test_indices for o in l]
				y_outer_test_preds = np.concatenate(y_outer_test_preds)
				outer_test_indices = np.concatenate(outer_test_indices)

			except TypeError: # LeaveOneOutCV
				print(y_outer_test_preds)
				print(outer_test_indices)
				#y_outer_test_preds = [l for l in y_outer_test_preds]
				#outer_test_indices = [l for l in outer_test_indices]
				pass

			df_preds = pd.DataFrame({'ind':outer_test_indices, 'pred':y_outer_test_preds})
			df_preds = df_preds.sort_values('ind')
			print(df_preds)
			y_outer_test_preds = df_preds['pred'].values
			print(df_preds)

			df_probas = pd.DataFrame(y_outer_test_probas)
			df_probas['ind'] = outer_test_indices
			print(df_probas)
			df_probas = df_probas.sort_values('ind')
			print(df_probas)
			y_outer_test_probas = df_probas.drop(['ind'], axis = 1).values

			for r in sorted(nestedcv_results, key = lambda x: x[0]):
				list_inner_best_params.append(r[4])

				if threshold:
					list_inner_best_thresholds.append(r[5])
				else:
					list_inner_best_thresholds = []

				list_inner_best_scores.append(r[6])

				if isinstance(set_inner_selected_features, set):
					set_inner_selected_features.update(r[7])
				else:
					print("issue hparams")
					print(hparams)
					print(set_inner_selected_features)

			outer_res_seed = calculate_random_seed_metrics(y, y_outer_test_preds, y_outer_test_probas, set_inner_selected_features, list_inner_best_params, list_inner_best_thresholds, list_inner_best_scores, base_clf_from_nestedcv_step, l_random_seeds[seed], directory_from_nestedcv_step, metric, p_grid_from_nestedcv_step)
			deployment_phase(p_grid_from_nestedcv_step, sample_weight, l_random_seeds[seed], base_clf_from_nestedcv_step, X, y, StratifiedKFold(n_splits = 10), directory_from_nestedcv_step, metric, target_name, feature_names, l_class_names, data, threshold)
			default_res_seed = default_model_performance(base_clf_from_nestedcv_step, X, y, sample_weight, StratifiedKFold(n_splits = 10), l_random_seeds[seed], p_grid_from_nestedcv_step, target_name, feature_names, l_class_names)
			permutation_res_seed = permutation_test_10cv(base_clf_from_nestedcv_step, X, y, sample_weight, l_random_seeds[seed])

			global_results.append((outer_res_seed, default_res_seed, permutation_res_seed))

	l_df_tuning = []
	l_df_default = []
	l_df_permut = []

	for r in global_results:
		l_df_tuning.append(r[0])
		l_df_default.append(r[1])
		l_df_permut.append(r[2])

	df_tuning = pd.concat(l_df_tuning)
	df_default = pd.concat(l_df_default)
	df_permutation = pd.concat(l_df_permut)

	df_default.to_csv(default_tsv, sep = '\t', index = False)

	path_checkpoint_defaults = "checkpoint_default_*"
	l_path_checkpoint_defaults = glob.glob(path_checkpoint_defaults)
	for c in l_path_checkpoint_defaults:
		os.remove(c)

	df_permutation.to_csv(permutation_tsv, sep = '\t', index = False)
	path_checkpoint_permut = "checkpoint_permutation_test_*"
	l_path_checkpoint_permut = glob.glob(path_checkpoint_permut)
	for c in l_path_checkpoint_permut:
		os.remove(c)

	df_tuning.to_csv(omc_tsv, sep = '\t', index = False)

	print(df_tuning)
	print(df_default)
	print(df_permutation)

	#####BOXPLOT###########
	# Draw boxplots (if nb seeds > 1, else swarmplot of outer test MCCs with standard deviation)
	if clf == 'CART':
		plot_filename_mcc = "hyperparameter_tuning_CART_MCC_metric.png"
		plot_filename_auc = "hyperparameter_tuning_CART_AUC_metric.png"
		plot_filename_aucpr = "hyperparameter_tuning_CART_AUCPR_metric.png"

	elif clf == 'PCART':
		plot_filename_mcc = "hyperparameter_tuning_FeatureSelection_CART_MCC_metric.png"
		plot_filename_auc = "hyperparameter_tuning_FeatureSelection_CART_AUC_metric.png"
		plot_filename_aucpr = "hyperparameter_tuning_FeatureSelection_CART_AUCPR_metric.png"

	boxplot_or_swarmplot_metric(clf, df_tuning, df_default, df_permutation, plot_filename_mcc, "MCC", nb_random_seeds, X.shape[0])
	boxplot_or_swarmplot_metric(clf, df_tuning, df_default, df_permutation, plot_filename_auc, "AUC", nb_random_seeds, X.shape[0])
	boxplot_or_swarmplot_metric(clf, df_tuning, df_default, df_permutation, plot_filename_aucpr, "AUCPR", nb_random_seeds, X.shape[0])
	end_date = time.strftime("%Y-%m-%d at %H:%M:%S")