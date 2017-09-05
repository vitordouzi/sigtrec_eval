# sigtrec_eval
sigtrec_eval is a [trec_eval](http://trec.nist.gov/trec_eval/) python-wrapper that get the output, process, and present in a clear and configurable format.

Requirements
------------
sigtrec_eval uses [numpy](http://www.numpy.org/), [scipy](https://www.scipy.org/) and [pandas](http://pandas.pydata.org/) to compute the output. To install those dependencies:
```
pip install -r requirements.txt
```
Input Format
------------
The input format is based on trec_eval input format.

qrel is the ground-truth file, which consists of each line with tuples of the form:

``qid  iter  docno  rel``

Read text tuples from trec_top_file of the form:
```
     qid iter   docno      rank  sim   run_id
ex.: 030  Q0  ZF08-175-870  0   4238   prise1
```

For more information, including measures availables and their descriptions, visit [trec_eval README](http://www-nlpir.nist.gov/projects/t01v/trecvid.tools/trec_eval_video/A.README).

Usage
------------
```
usage: sigtrec_eval.py [-h] [-m M [M ...]] [-t [T]]
  qrel baseline_result
  [result_to_compare [result_to_compare ...]]
  [-s [{ttest,welchttest,None} [{ttest,welchttest,None} ...]]]
  [-f [{csv,html,json,latex,sql,string}]] [-o [O]]
positional arguments:
  qrel                  qrel file in trec_eval format
  baseline_result       The baseline result to evaluate
  result_to_compare     The results to compare with the baseline
optional arguments:
  -h, --help            show this help message and exit
  -m M [M ...]          Evaluation method
  -t [T]                The trec_eval executor path
  -s [{ttest,welchttest,None} [{ttest,welchttest,None} ...]]
                        Statistical test
  -f [{csv,html,json,latex,sql,string}]
                        Output format
  -o [O]                Output file
```

Example
------------

Compute precision@10:
```
$ python3 sigtrec_eval.py example/qrelFile.qrel example/baseline example/result_to_compare1 -m P.10
             Approach       P_10
0            baseline  0.1960 bl
1  result_to_compare1    0.2071
```

Compute precision@10 and recall@10:
```
$ python3 sigtrec_eval.py example/qrelFile.qrel example/baseline example/result_to_compare1 -m P.10 recall.10
             Approach       P_10  recall_10
0            baseline  0.1960 bl  0.1669 bl
1  result_to_compare1    0.2071     0.1711
```

Using latex output format:
```
$ python3 sigtrec_eval.py example/qrelFile.qrel example/baseline example/result_to_compare1 -m ndcg_cut.10 map_cut.10 -f latex
\begin{tabular}{llll}
\toprule
{} &            Approach & ndcg\_cut\_10 & map\_cut\_10 \\
\midrule
0 &            baseline &   0.2482 bl &  0.0979 bl \\
1 &  result\_to\_compare1 &     0.2231  &    0.0805  \\
\bottomrule
\end{tabular}
```

Generate t-studdent test:
```
$ python3 sigtrec_eval.py example/qrelFile.qrel example/baseline example/result_to_compare1 example/result_to_compare2 -m P.10 recip_rank -s ttest
             Approach       P_10 recip_rank
0            baseline  0.1960 bl  0.5467 bl
1  result_to_compare1   0.2071 ▲   0.4004 ▼
2  result_to_compare2   0.0002 ▼   0.0529 ▼
```

Save the output into a file:
```
$ python3 sigtrec_eval.py example/qrelFile.qrel example/baseline example/result_to_compare1 -m Rprec bpref -f csv -o output.csv
$ cat output.csv 
,Approach,Rprec,bpref
0,baseline,0.1832 bl,0.2418 bl
1,result_to_compare1,0.2031 ,0.2787
```