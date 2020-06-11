# -*- coding: utf-8 -*-
"""
We collect custom scoring functions
"""

from sklearn.metrics import make_scorer, confusion_matrix, roc_auc_score
def gini(solution, submission):
	df = zip(solution, submission, range(len(solution)))
	df = sorted(df, key=lambda x: (x[1],-x[2]), reverse=True)
	rand = [float(i+1)/float(len(df)) for i in range(len(df))]
	totalPos = float(sum([x[0] for x in df]))
	cumPosFound = [df[0][0]]
	for i in range(1,len(df)):
		cumPosFound.append(cumPosFound[len(cumPosFound)-1] + df[i][0])
	Lorentz = [float(x)/totalPos for x in cumPosFound]
	Gini = [Lorentz[i]-rand[i] for i in range(len(df))]
	return sum(Gini)/len(df)
def normalized_gini(solution, submission, gini=gini):
    "most likely this computes Somer's D which is widely known as the Gini coefficient"
    normalized_gini = gini(solution, submission)/gini(solution, solution)
    return normalized_gini

def gini_normalized_new(y_actual, y_pred):
    """Simple normalized Gini based on Scikit-Learn's roc_auc_score"""
    
    # If the predictions y_pred are binary class probabilities
    if y_pred.ndim == 2:
        if y_pred.shape[1] == 2:
            y_pred = y_pred[:, 1]
    gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
    return gini(y_actual, y_pred) / gini(y_actual, y_actual)
def custom_score_function(y_true, y_pred):
	conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
	tn, fp, fn, tp = conf_matrix.ravel()
	print(f'tn={tn}, fp = {fp}, fn = {fn}, tp = {tp} \n')
	
	gini_value = normalized_gini(solution=y_true, submission=y_pred)
	abs_gini = abs(gini_value)
	if abs_gini < 0:
		print('less than zero')
	print('gini = {}'.format(abs_gini))
	roc_auc_value = roc_auc_score(y_true=y_true, y_score=y_pred)
	
	return abs_gini
    
gini_scorer = make_scorer(normalized_gini, greater_is_better = True)
new_gini_scorer = make_scorer(gini_normalized_new, greater_is_better = True, needs_proba=True)
custom_scorer = make_scorer(custom_score_function, greater_is_better=True)
