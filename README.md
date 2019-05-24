# SemBleu: A Robust Metric for AMR Parsing Evaluation

The repository corresponds to our recent ACL 2019 paper entitled "SemBleu: A Robust Metric for AMR Parsing Evaluation".

## Usage

```
chmod a+x eval.sh
./eval.sh output-file-path reference-file-path
```

Same as Smatch, AMRs in each file are seperated by one empty line, such as:

```
(a / ask-01 :ARG0 (b / boy) :ARG1 (q / question))

(a / answer-01 :ARG0 (g / girl) :ARG1 (q / question))

```
