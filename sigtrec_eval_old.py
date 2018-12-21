# -*- coding: utf-8 -*-
"""
Created on Mon Out 1 10:07:00 2017

@author: Vítor Mangaravite
"""
import sys
import os
import subprocess
import argparse
import numpy as np
import pandas as pd
import multiprocessing
from sklearn.model_selection import KFold
from scipy.stats.mstats import ttest_rel
from scipy.stats import ttest_ind, wilcoxon
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE

class InputAction(argparse.Action):
    def __init__(self, *args, **kwargs):
        super(InputAction, self).__init__(*args, **kwargs)
        self.nargs = '+'
    def __call__(self, parser, namespace, values, option_string):
        lst = getattr(namespace, self.dest, []) or []
        if len(values) > 1:
        	lst.append(InputResult(values[0], values[1:]))
        else:
        	with open(values[0].name) as file_input:
        		for line in file_input.readlines():
        			parts = line.strip().split(' ')
        			lst.append(InputResult(parts[0], parts[1:]))
        setattr(namespace, self.dest, lst)

class InputResult(object):
	def __init__(self, qrel, result_to_compare):
		if type(qrel) is str: 
			self.qrel = qrel
			self.result_to_compare = result_to_compare
		else:
			self.qrel = qrel.name
			self.result_to_compare = [ x.name for x in result_to_compare ]
	def __repr__(self):
		return 'InputResult(%r, %r)' % (self.qrel, self.result_to_compare)

def get_test(test, pbase, pcomp, multi_test=False):
	if np.array_equal(pbase.values, pcomp.values):
		pvalue = 1.
	else:
		if test == 'student':
			(tvalue, pvalue) = ttest_rel(pbase, pcomp)
		elif test == 'wilcoxon':
			(tvalue, pvalue) = wilcoxon(pbase, pcomp)
		elif test == 'welcht':
			(tvalue, pvalue) = ttest_ind(pbase, pcomp, equal_var=False)
	if pvalue < 0.05:
		pbase_mean = pbase.mean()
		pcomp_mean = pcomp.mean()
		if pvalue < 0.01:
			if pbase_mean > pcomp_mean:
				result_test = '▼ '
			else:
				result_test = '▲ '
		else:
			if pbase_mean > pcomp_mean:
				result_test = 'ᐁ '
			else:
				result_test = 'ᐃ '
	else:
		if not multi_test:
			result_test = '  '
		else:
			result_test = '⏺ '
	return result_test
def getQrel(qrelFile):
	return os.path.basename(qrelFile).replace('.qrel','')
def getApproach(nameFile):
	return os.path.basename(nameFile).replace('.out','')

def get_sampler(summ, random_state=0):
	if summ == "smote":
		return SMOTE(n_jobs=multiprocessing.cpu_count(), k_neighbors=10, random_state=random_state)
	return RandomOverSampler(random_state=random_state)

dir_trec_ = os.path.join(".", os.path.dirname(__file__), "trec_eval")
""" Configuring the argument parser """
parser = argparse.ArgumentParser()
parser.add_argument('-i','--input', action=InputAction, type=argparse.FileType('rt'), nargs='+', metavar='QREL BASELINE [TO_COMPARE ...]', help='The list of positional argument where the first arg is the qrel file, the second is the baseline result and the third is the optional list of results to compare.')
parser.add_argument('-m','--measure', type=str, nargs='+', help='Evaluation method.', default=['P.10', 'recall.10'])
parser.add_argument('-t','--trec_eval', type=str, nargs='?', help='The trec_eval executor path (Default: %s).' % dir_trec_, metavar='TREC_EVAL_PATH', default=dir_trec_)
parser.add_argument('-sum','--summary', type=str, nargs='*', help='Summary each approach results using Over-sampling methods (Default: None).', default=['None'], choices=['ros','smote'])
parser.add_argument('-s','--statistical_test', type=str, nargs='*', help='Statistical test (Default: student).', default=['student'], choices=['None','student','wilcoxon','welcht'])
parser.add_argument('-f','--format', type=str, nargs='?', help='Output format.', default='string', choices=['csv', 'html', 'json', 'latex', 'sql', 'string'])
parser.add_argument('-o','--output', type=str, nargs='?', help='Output file.', default='')
parser.add_argument('-cv','--cross-validation', type=int, nargs='?', help='Cross-Validation.', default=1)
parser.add_argument('-r','--round', type=int, nargs='?', help='Round the result.', default=4)

args = parser.parse_args()
args.cross_validation = max(1, args.cross_validation)
args.statistical_test = [st for st in args.statistical_test if st != 'None']
args.summary = [summ for summ in args.summary if summ != 'None']
print_summ = len(args.summary) > 0 and len(args.input) > 0 

"""Building DataFrames"""
df_raw = pd.DataFrame()
raw = []
for input_result in args.input:
	for m in args.measure:
		for (idx, to_compare) in enumerate(input_result.result_to_compare):
			content = str(subprocess.Popen(' '.join([args.trec_eval, input_result.qrel, to_compare, '-q', '-m', m]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1]
			raw.extend([ (getQrel(input_result.qrel), idx, getApproach(to_compare), *( w.strip() for w in line.split('\\t') )) for line in content.split('\\n') ][:-1])
df_raw = pd.DataFrame(raw, columns=['qrel', 'idx_approach', 'approach', 'measure', 'docid', 'result'])

df_finale = pd.pivot_table(df_raw, index=['qrel', 'idx_approach', 'approach', 'docid'], columns='measure', values='result', aggfunc='first')
df_finale[np.array(df_finale.columns)] = df_finale[np.array(df_finale.columns)].astype(np.float64)
evaluation_methods = list(df_finale.columns.values)

columns_names = ['app_name', *evaluation_methods, *(len(args.statistical_test)*evaluation_methods)]

multicolumn_names = list(zip(len(evaluation_methods)*['result'], evaluation_methods))
for test_name in args.statistical_test:
	multicolumn_names.extend( list(zip(len(evaluation_methods)*[test_name], evaluation_methods)) )

 
""" Process and print individual result by dataset """
for (qrel, qrel_group) in df_finale.groupby('qrel'):
	df_baseline = qrel_group.loc[qrel_group.index.get_level_values('idx_approach') == 0]
	if args.cross_validation > 1:
		kf = KFold(n_splits=args.cross_validation, shuffle=True, random_state=42)
		docids_test_folds = [ test_index for train_index, test_index in kf.split(df_baseline.index.get_level_values('docid')) ]
	raw = []
	for ((idx_app, app_name), app_group) in qrel_group.groupby(['idx_approach', 'approach']):
		line = [ app_name ]
		for (test, method) in multicolumn_names:
			if test == 'result':
				if args.cross_validation > 1:
					line.append( np.mean([ app_group[method][test_index].mean() for test_index in docids_test_folds ]) )
				else:
					line.append( app_group[method].mean() )
			elif idx_app == 0:
				line.append( 'bl' )
			else:
				pbase = df_baseline.loc[df_baseline.index.get_level_values('docid').isin(app_group.index.get_level_values('docid'))][method].sort_index() # Get the baseline result
				pcomp = app_group.loc[app_group.index.get_level_values('docid').isin(df_baseline.index.get_level_values('docid'))][method].sort_index()   # Get the results to compare
				line.append( get_test(test, pbase, pcomp, len(args.statistical_test) > 1) )
		raw.append(line)
	df_to_print = pd.DataFrame(raw, columns=columns_names)
	df_to_print = df_to_print.set_index('app_name')
	df_to_print.columns = pd.MultiIndex.from_tuples(multicolumn_names)

	df_to_print_r = df_to_print['result'].round(args.round).astype(str)
	for col in df_to_print.columns.levels[0][1:]:
		df_to_print_r = df_to_print_r + ' ' + df_to_print[col].astype(str)

	with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
		print(getattr(df_to_print_r, 'to_'+args.format)())
	print()
if print_summ:
	""" Process and print the results summarization """
	df_finale = pd.pivot_table(df_raw, index=['qrel', 'docid'], columns=['idx_approach','measure'], values='result', aggfunc='first').dropna()
	df_finale[np.array(df_finale.columns)] = df_finale[np.array(df_finale.columns)].apply(pd.to_numeric)
	for summ in args.summary:
		y = pd.factorize(df_finale.index.get_level_values('qrel'))[0]
		raw = []
		for repetition in range(args.cross_validation):
			sampler = get_sampler(summ, random_state=repetition)
			X_res, y_res = sampler.fit_sample(df_finale, y)
			raw.extend(X_res)
		df_result = pd.DataFrame(raw, columns=df_finale.columns).round(args.round)
		raw = []
		df_baseline = df_result[0]
		print(list(zip(df_result[0]['P_10'][df_result[0]['P_10']!=df_result[2]['P_10']],df_result[2]['P_10'][df_result[0]['P_10']!=df_result[2]['P_10']])))
		for idx_app in df_result.columns.levels[0]:
			line = [ idx_app ]
			for (test, method) in multicolumn_names:
				if test == 'result':
					line.append( df_result[idx_app][method].mean() )
				elif idx_app == 0:
					line.append( 'bl' )
				else:
					pbase = df_baseline[method] # Get the baseline result
					pcomp = df_result[idx_app][method]   # Get the results to compare
					line.append( get_test(test, pbase, pcomp, len(args.statistical_test) > 1) )
			raw.append(line)
		df_to_print = pd.DataFrame(raw, columns=columns_names)

		df_to_print = df_to_print.set_index('app_name')
		df_to_print.columns = pd.MultiIndex.from_tuples(multicolumn_names)
		df_to_print_r = df_to_print['result'].round(args.round).astype(str)
		for col in df_to_print.columns.levels[0][1:]:
			df_to_print_r = df_to_print_r + ' ' + df_to_print[col].astype(str)
		
		print(sampler)
		with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
			print(getattr(df_to_print_r, 'to_'+args.format)())
		print()