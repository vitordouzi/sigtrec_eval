# sigtrec_eval
[trec_eval](http://trec.nist.gov/trec_eval/) python wrapper.

Requirements
------------


Usage
------------
	``usage: sigtrec_eval.py [-h] [-m M [M ...]] [-t [T]]
    [-s [{ttest,welchttest,None} [{ttest,welchttest,None} ...]]]
    [-f [{csv,html,json,latex,sql,string}]] [-o [O]]
    qrel baseline_result
    [result_to_compare [result_to_compare ...]]
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
    -o [O]                Output to save result``
