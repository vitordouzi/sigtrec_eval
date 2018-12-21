import sys, os, subprocess, math, multiprocessing, random
import numpy as np
from numpy import nan
import pandas as pd
from scipy.stats.mstats import ttest_rel
from scipy.stats import ttest_ind, wilcoxon
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import namedtuple

Result = namedtuple('Result', ['qrelFileName', 'datasetid', 'resultsFiles', 'nameApproach'])

def getFileName(qrelFile):
	return os.path.basename(qrelFile)

class SIGTREC_Eval():
	def __init__(self, cv=0, seed=42, round_=4, trec_eval=os.path.join(".", os.path.dirname(__file__), "trec_eval")):
		self.nameApp = {}
		self.cv = cv
		self.seed = seed
		self.trec_eval = trec_eval
		self.round = round_
		random.seed(seed)
	def _build_F1(self, qrelFileName, to_compare, m, top):
		command = ' '.join([self.trec_eval, qrelFileName, to_compare, '-q ', '-M %d' % top, '-m %s.%d'])
		content_P = str(subprocess.Popen(command % ('P', top), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
		content_R = str(subprocess.Popen(command % ('recall', top), stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
		content_F1 = []
		for i in range(len(content_P)):
			part_P = content_P[i].split('\\t')
			part_R = content_R[i].split('\\t')
			if len(part_P) != len(part_R) or len(part_P) < 3:
				continue
			if part_P[1] != part_R[1]:
				print(part_P[1], part_R[1])
			else:
				Pre = float(part_P[2])
				Rec = float(part_R[2])
				if Pre == 0. or Rec == 0.:
					content_F1.append( 'F1_%d\\t%s\\t0.' % (  top, part_P[1] ) )
				else:
					line = 'F1_%d\\t%s\\t%.4f' % ( top, part_P[1], (2.*Pre*Rec)/(Pre+Rec) )
					content_F1.append( line )
		return content_F1
	def build_df(self, results, measures, top):
		raw = []
		qtd = len(measures)*sum([ len(input_result.resultsFiles) for input_result in results])
		i=0
		for input_result in results:
			self.nameApp[input_result.datasetid] = []
			for m in measures:
				for (idx, to_compare) in enumerate(input_result.resultsFiles):
					self.nameApp[input_result.datasetid].append(getFileName(to_compare))
					print("\r%.2f%%" % (100.*i/qtd),end='')
					i+=1
					if m.startswith("F1"):
						content = self._build_F1(input_result.qrelFileName, to_compare, m, top=top)
					else:
					############################################################################################## Tamanho 10 FIXADO ##############################################################################################
						command = ' '.join([self.trec_eval, input_result.qrelFileName, to_compare, '-q -M %d' % top, '-m', m])
						content = str(subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True).communicate()[0])[2:-1].split('\\n')
					raw.extend([ (input_result.datasetid, idx, getFileName(to_compare), *( w.strip() for w in line.split('\\t') )) for line in content ][:-1])
		print("\r100.00%%")
		df_raw = pd.DataFrame(list(filter(lambda x: not x[4]=='all', raw)), columns=['qrel', 'idx_approach', 'approach', 'measure', 'docid', 'result'])
		df_finale = pd.pivot_table(df_raw, index=['qrel', 'docid'], columns=['idx_approach','measure'], values='result', aggfunc='first')
		df_finale.reset_index()
		df_finale[np.array(df_finale.columns)] = df_finale[np.array(df_finale.columns)].astype(np.float64)
		df_finale.replace('None', 0.0, inplace=True)
		df_finale.replace(nan, 0.0, inplace=True)
		#df_finale = df_finale[~df_finale['docid'].isin(['all'])]
		df_finale['fold'] = [0]*len(df_finale)
		if self.cv > 0:
			for (qrel, qrel_group) in df_finale.groupby('qrel'):
				folds=(list(range(self.cv))*math.ceil(len(qrel_group)/self.cv))[:len(qrel_group)]
				random.shuffle(folds)
				df_finale.loc[qrel, 'fold'] = folds
		#with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
		#	print(df_finale)
		return df_finale
	def get_test(self, test, pbase, pcomp, multi_test=False):
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
	def build_printable(self, table, significance_tests):
		printable = {}
		for qrel, qrel_group in table.groupby('qrel'):
			raw = []
			base = qrel_group.loc[:,0]
			for idx_app in [idx for idx in qrel_group.columns.levels[0] if type(idx) == int]:
				instance = [ self.nameApp[qrel][idx_app] ]
				for m in qrel_group[idx_app].columns:
					array_results = qrel_group[idx_app][m]
					#print(qrel_group.groupby('fold').mean()[idx_app][m])
					mean_measure_folds = qrel_group.groupby('fold').mean()[idx_app][m].mean()
					test_result=""
					for test in significance_tests:
						if idx_app > 0:
							test_result+=(self.get_test(test, base[m], array_results, len(significance_tests)>1))
						else:
							test_result+=('bl ')
					instance.append('%f %s' % (round(mean_measure_folds,self.round), test_result) )
				raw.append(instance)
			printable[qrel] = pd.DataFrame(raw, columns=['app', *(table.columns.levels[1].get_values())[:-1]])
		return printable
	def get_sampler(self,  sampler_name):
		if sampler_name == "ros" or sampler_name == 'RandomOverSampler':
			return RandomOverSampler(random_state=self.seed)
		if sampler_name == "SMOTE" or sampler_name == "smote":
			return SMOTE(random_state=self.seed)
	def build_over_sample(self, df, sampler):
		raw = []
		for fold, fold_group in df.groupby('fold'):
			y = pd.factorize(fold_group.index.get_level_values('qrel'))[0]
			X_sampled, y_res = sampler.fit_sample(fold_group, y)
			raw.extend(X_sampled)
		df_sampled = pd.DataFrame(raw, columns=df.columns)
		df_sampled['qrel'] = [sampler.__class__.__name__]*len(df_sampled)
		self.nameApp[sampler.__class__.__name__] = self.nameApp[list(self.nameApp.keys())[0]]
		return df_sampled
