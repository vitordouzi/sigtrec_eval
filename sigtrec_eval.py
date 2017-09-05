# -*- coding: utf-8 -*-
"""
Created on Fri Set 1 16:07:00 2017

@author: Vítor Mangaravite
"""
import sys
import os
import subprocess
import argparse
import numpy as np
import pandas as pd
from scipy.stats.mstats import ttest_rel
from scipy.stats import ttest_ind

class result(object):
	def __init__(self, metric, k=None):
		self.metric = metric
		self.k = k
		self.baseline = 0.0
		self.avg_result = 0.0
		self.tests_pvalue = {}
		self.results = {}
	def print_tests(self, _format):
		s = ' '
		for (t, v) in self.tests_pvalue.items():
			#{ 'GT_TEST_01':'▲', 'LT_TEST_01':'▼', 'GT_TEST_05':'ᐃ', 'LT_TEST_05':'ᐁ' }
			if v < 0.01:
				if self.avg_result < self.baseline:
					s += _format['LT_TEST_01']
				elif self.avg_result > self.baseline:
					s += _format['GT_TEST_01']
			elif v < 0.05:
				if self.avg_result < self.baseline:
					s += _format['LT_TEST_05']
				elif self.avg_result > self.baseline:
					s += _format['GT_TEST_05']
			else:
				s += ' '
		return s

	def _add_instance(self, doc_id, value):
		self.results[doc_id] = float(value)
	def _statistical_test(self, other_result, tests):
		for test_name in tests:
			self.tests_pvalue[test_name] = getattr(self, '_test_'+test_name)(other_result)
	def avg(self):
		self.avg_result = np.mean( [ v for (k,v) in self.results.items() ] )
	def _test_ttest(self, other_result):
		doc_ids = set( self.results.keys() & other_result.results.keys() )
		pbase, pcomp = [ self.results[doc_id] for doc_id in doc_ids ], [ other_result.results[doc_id] for doc_id in doc_ids ]
		if pbase != pcomp:
			(tvalue, pvalue) = ttest_rel(pbase, pcomp)
			return pvalue
		return 1.
	def _test_welchttest(self, other_result):
		doc_ids = set( self.results.keys() & other_result.results.keys() )
		pbase, pcomp = [ self.results[doc_id] for doc_id in doc_ids ], [ other_result.results[doc_id] for doc_id in doc_ids ]
		if pbase != pcomp:
			(tvalue, pvalue) = ttest_ind(pbase, pcomp, equal_var=False)
			return pvalue
		return 1.
class resultSet(object):
	def __init__(self, approach, base=None):
		self.approach = approach
		self.base = base
		self.results_metrics = {}
	def add_resultset(self, trec_result_):
		result_raw = [ [ w.strip() for w in line.split('\\t') ] for line in trec_result_.split('\\n') ]
		for ( metric, doc_id, value ) in result_raw[:-1]:
			if doc_id != 'all':
				if metric not in self.results_metrics:
					(metric_name, k) = self._get_metric(metric)
					self.results_metrics[metric] = result(metric_name, k=k)
				self.results_metrics[metric]._add_instance(doc_id, value)
	def _get_metric(self, metric):
		parts = metric.split('_')
		if parts[-1].isnumeric():
			return ( '_'.join(parts[:-1]), int(parts[-1]) )
		return (metric, None)
	def avg(self):
		for m in self.results_metrics:
			self.results_metrics[m].avg()
			if self.base == None:
				self.results_metrics[m].baseline = self.results_metrics[m].avg_result
			else:
				self.results_metrics[m].baseline = self.base.results_metrics[m].avg_result
	def statistical_test(self, other_resultset, tests):
		for m in self.results_metrics:
			if m in other_resultset.results_metrics:
				self.results_metrics[m]._statistical_test( other_resultset.results_metrics[m], tests )

def print_dataframe(base, list_to_compare, format_to_print, _format='string'):
	df = pd.DataFrame()
	approach_colum = [ base.approach ]
	list(map( lambda x: approach_colum.append(x.approach), list_to_compare))
	df['Approach'] = approach_colum
	metrics = list(base.results_metrics.keys())
	for m in metrics:
		approach_colum = [ '%.4f bl' % base.results_metrics[m].avg_result ]
		list(map( lambda x: approach_colum.append('%.4f%s' % (x.results_metrics[m].avg_result, x.results_metrics[m].print_tests(format_to_print) ) ), list_to_compare))
		df[m] = approach_colum
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
		return getattr(df, 'to_'+_format)()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='Sigeval A/B Testing.')

	choices_s = [str(func).replace("_test_",'') for func in dir(result) if callable(getattr(result, func)) and func.startswith("_test_")]
	choices_s.append('None')
	parser.add_argument('qrel', type=str, nargs=1, help='qrel file in trec_eval format')
	parser.add_argument('baseline_result', type=str, nargs=1, help='The baseline result to evaluate')
	parser.add_argument('result_to_compare', type=str, nargs='*', help='The results to compare with the baseline')
	parser.add_argument('-m', type=str, nargs='+', help='Evaluation method')
	parser.add_argument('-t', type=str, nargs='?', help='The trec_eval executor path', default='./trec_eval')
	parser.add_argument('-s', type=str, nargs='*', help='Statistical test', default=['None'], choices=choices_s)
	parser.add_argument('-f', type=str, nargs='?', help='Output format', default='string', choices=['csv', 'html', 'json', 'latex', 'sql', 'string'])
	parser.add_argument('-o', type=str, nargs='?', help='Output file', default='')

	args = parser.parse_args()

	tests = [ s for s in args.s if s != 'None']

	#format_to_print = {}
	#format_to_print['to_string'] = { 'GT_TEST_01':'▲', 'LT_TEST_01':'▼', 'GT_TEST_05':'ᐃ', 'LT_TEST_05':'ᐁ' }
	#format_to_print['to_latex'] = { 'GT_TEST_01':'▲', 'LT_TEST_01':'▼', 'GT_TEST_05':'ᐃ', 'LT_TEST_05':'ᐁ' }
	#format_to_print['to_html'] = { 'GT_TEST_01':'▲', 'LT_TEST_01':'▼', 'GT_TEST_05':'ᐃ', 'LT_TEST_05':'ᐁ' }

	format_to_print = { 'GT_TEST_01':'▲', 'LT_TEST_01':'▼', 'GT_TEST_05':'ᐃ', 'LT_TEST_05':'ᐁ' }


	results_to_compare = {}
	baseline_resultset = None
	for m in args.m:
		baseline_id = os.path.basename(args.baseline_result[0])
		if baseline_resultset == None:
			baseline_resultset = resultSet(baseline_id)
		p = subprocess.Popen(' '.join([args.t, args.qrel[0], args.baseline_result[0], '-q', '-m', m]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
		outc, err = p.communicate()
		baseline_resultset.add_resultset(str(outc)[2:-1])
		for to_compare in args.result_to_compare:
			to_compare_id = os.path.basename(to_compare)
			if to_compare_id not in results_to_compare:
				results_to_compare[to_compare_id] = resultSet(to_compare_id, baseline_resultset)
			p = subprocess.Popen(' '.join([args.t, args.qrel[0], to_compare, '-q', '-m', m]), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
			outc, err = p.communicate()
			results_to_compare[to_compare_id].add_resultset(str(outc)[2:-1])
	baseline_resultset.avg()
	for to_compare in results_to_compare.values():
		to_compare.avg()
		to_compare.statistical_test( baseline_resultset, tests )
	if args.o == '':
		print(print_dataframe(baseline_resultset, results_to_compare.values(), format_to_print, _format=args.f))
	else:
		with open(args.o, 'w') as fil:
			fil.write(print_dataframe(baseline_resultset, results_to_compare.values(), format_to_print, _format=args.f))