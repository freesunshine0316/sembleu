
import os, sys, json, time
from bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction, NgramInst
from amr_graph import AMRGraph

def read_amr(path):
    ids = []
    id_dict = {}
    amrs = []
    amr_str = ''
    for line in open(path,'rU'):
        if line.startswith('#'):
            if line.startswith('# ::id'):
                id = line.strip().split()[2]
                ids.append(id)
                id_dict[id] = len(ids)-1
            continue
        line = line.strip()
        if line == '':
            if amr_str != '':
                amrs.append(amr_str.strip())
                amr_str = ''
        else:
            amr_str = amr_str + line + ' '

    if amr_str != '':
        amrs.append(amr_str.strip())
        amr_str = ''
    return amrs

def get_amr_ngrams(path, stat_save_path=None):
    data = []
    if stat_save_path:
        f = open(stat_save_path, 'w')
    for line in read_amr(path):
        try:
            amr = AMRGraph(line.strip())
        except AssertionError:
            print line
            assert False
        amr.revert_of_edges()
        ngrams = amr.extract_ngrams(3, multi_roots=True) # dict(list(tuple))
        data.append(NgramInst(ngram=ngrams, length=len(amr.edges)))
        if stat_save_path:
            print >>f, len(amr), len(ngrams[1]), len(ngrams[2]), len(ngrams[3])
    if stat_save_path:
        f.close()
    return data

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print 'python this-script ans-file ref-file'
        sys.exit(0)
    print 'loading ...'
    hypothesis = get_amr_ngrams(sys.argv[1])
    references = [[x] for x in get_amr_ngrams(sys.argv[2])]
    smoofunc = getattr(SmoothingFunction(), 'method3')
    print 'evaluating ...'
    st = time.time()
    if len(sys.argv) == 4:
        n = int(sys.argv[3])
        weights = (1.0/n, )*n
    else:
        weights = (0.34, 0.33, 0.34)
    print corpus_bleu(references, hypothesis, weights=weights, smoothing_function=smoofunc, auto_reweigh=True)
    print 'time:', time.time()-st, 'secs'
