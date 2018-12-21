
import argparse, os
from sigtrec_eval import SIGTREC_Eval
from collections import namedtuple
import pandas as pd

def getFileName(qrelFile):
    return os.path.basename(qrelFile)

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

Result = namedtuple('Result', ['qrelFileName', 'datasetid', 'resultsFiles', 'nameApproach'])
dir_trec_ = os.path.join(".", os.path.dirname(__file__), "trec_eval")
""" Configuring the argument parser """
parser = argparse.ArgumentParser()
parser.add_argument('-I','--input_file', type=str, nargs='*', help='Input as a file of list of QREL BASELINE [TO_COMPARE ...].')
parser.add_argument('-i','--input', action=InputAction, type=argparse.FileType('rt'), nargs='*', metavar='QREL BASELINE [TO_COMPARE ...]', help='The list of positional argument where the first arg is the qrel file, the second is the baseline result and the third is the optional list of results to compare.')
parser.add_argument('-m','--measure', type=str, nargs='+', help='Evaluation method.', default=['P.10', 'recall.10'])
parser.add_argument('-t','--trec_eval', type=str, nargs='?', help='The trec_eval executor path (Default: %s).' % dir_trec_, metavar='TREC_EVAL_PATH', default=dir_trec_)
parser.add_argument('-sum','--summary', type=str, nargs='*', help='Summary each approach results using Over-sampling methods (Default: None).', default=['None'], choices=['ros','smote'])
parser.add_argument('-s','--statistical_test', type=str, nargs='*', help='Statistical test (Default: None).', default=['None'], choices=['student','wilcoxon','welcht'])
parser.add_argument('-f','--format', type=str, nargs='?', help='Output format.', default='string', choices=['csv', 'html', 'json', 'latex', 'sql', 'string'])
parser.add_argument('-o','--output', type=str, nargs='?', help='Output file.', default='')
parser.add_argument('-cv','--cross-validation', type=int, nargs='?', help='Cross-Validation.', default=1)
parser.add_argument('-r','--round', type=int, nargs='?', help='Round the result.', default=4)
parser.add_argument('-T','--top',type=int, nargs='?', help='Size of the rank.', default=10)

args = parser.parse_args()

statistical_test = [ st for st in args.statistical_test if st != "None"]

results = []
if args.input_file != None:	
	for input_file in args.input_file:
		with open(input_file) as fil:
			for line in fil:
				parts = line.replace('\n','').split(' ')
				results.append(Result(qrelFileName=parts[0], datasetid=getFileName(parts[0]), resultsFiles=parts[1:], nameApproach=[]))
if args.input != None:
	for input_result in args.input:
		results.append(Result(qrelFileName=input_result.qrel, datasetid=getFileName(input_result.qrel), resultsFiles=input_result.result_to_compare, nameApproach=[]))
# cv=0, seed=42, trec_eval='./trec_eval'
sig = SIGTREC_Eval(cv = args.cross_validation, seed=42, trec_eval=args.trec_eval, round_=args.round)

df = sig.build_df(results, top=args.top, measures=args.measure)
args.summary = [ test for test in args.summary if test != 'None']
printable = sig.build_printable(df, statistical_test)

for qrel in printable:
	with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
		print(getattr(printable[qrel], 'to_'+args.format)())

for name_sampler in args.summary:
	sampler = sig.get_sampler(name_sampler)
	sampled = sig.build_over_sample(df, sampler)
	printable = sig.build_printable(sampled, statistical_test)
	for qrel in printable:
		with pd.option_context('display.max_rows', None, 'display.max_columns', 10000000000):
			print(getattr(printable[qrel], 'to_'+args.format)())
