from math import log, log10, exp

LOGZERO = -100

def eexp(x):
    if x <= LOGZERO:
        return 0
    else:
        return exp(x)

def elog(x):
    if x == 0:
        return LOGZERO
    else:
        return log(x)

def elog10(x):
    if x == 0:
        return LOGZERO
    else:
        return log10(x)

def logsum(logx, logy):
    if logx <= LOGZERO or logy <= LOGZERO:
        if logx<= LOGZERO and logy <= LOGZERO:
            return LOGZERO
        if logx <= LOGZERO:
            return logy
        else:
            return logx
    else:
        if logx > logy:
            return logx + log(1 + exp(logy - logx))
        else:
            return logy + log(1 + exp(logx - logy))

def logprod(logx, logy):
    if logx <= LOGZERO or logy <= LOGZERO:
        return LOGZERO
    else:
        return logx + logy
