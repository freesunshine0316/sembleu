# SemBleu: A Robust Metric for AMR Parsing Evaluation

The repository corresponds to our recent ACL 2019 paper entitled "SemBleu: A Robust Metric for AMR Parsing Evaluation".

## Results

The table below lists the SemBleu scores of recent SOTA work. The numbers are obtained by running our script on their provided outputs.

| Model | SemBleu |
|---|---|
| LDC2015E86 ||
| [Lyu and Titov, (ACL 2018)](https://www.aclweb.org/anthology/P18-1037) | 58.7 |
| [Groschwitz et al., (ACL 2018)](https://www.aclweb.org/anthology/P18-1170) | 51.8 |
| [Guo and Lu, (EMNLP 2018)](https://www.aclweb.org/anthology/D18-1198) | 50.4 |
| LDC2016E25 ||
| [Lyu and Titov, (ACL 2018)](https://www.aclweb.org/anthology/P18-1037) | 60.3 |
| [van Noord and Bos, (CLIN 2017)](https://arxiv.org/abs/1705.09980) | 49.5 |
| LDC2017T10 ||
| [Zhang et al., (ACL 2019)](https://www.aclweb.org/anthology/P19-1009) | 59.9 |
| [Cai and Lam (EMNLP 2019)]() | 56.9 |
| [Groschwitz et al., (ACL 2018)](https://www.aclweb.org/anthology/P18-1170) | 52.5 |
| [Guo and Lu, (EMNLP 2018)](https://www.aclweb.org/anthology/D18-1198) | 52.4 |

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
