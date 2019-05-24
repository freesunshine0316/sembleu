import logger
from rule import isvar
import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_list(
    'lex',
    None,
    'Lexical weight files (lex.e2f, lex.f2e).')

class LexicalWeighter(object):
    def __init__(self,
                 fweightfile=None,
                 eweightfile=None,
                 ratiofile=None):
        self.fweighttable = None
        self.eweighttable = None
        self.ratiotable = None
        # override parameters passed in by __init__ with
        # FLAGS for compatibility for now
        if FLAGS.lex is not None:
            fweightfile = FLAGS.lex[0]
            eweightfile = FLAGS.lex[1]

        if fweightfile is not None:
            self.fweighttable = read_weightfile(open(fweightfile))
        if eweightfile is not None:
            self.eweighttable = read_weightfile(open(eweightfile))
        if ratiofile is not None:
            self.ratiotable = read_weightfile(open(ratiofile))
            for ((word1,word2),p) in self.ratiotable.iteritems():
                # this cutoff determined by looking at ratio vs. rank
                # basically the curve has three parts:
                if p > 100.0:
                    self.ratiotable[word1,word2] = 1.0
                elif p > 1.0: # long flat region
                    self.ratiotable[word1,word2] = 0.5
                else:
                    self.ratiotable[word1,word2] = 0.0

    def compute_lexical_weights(self, a):
        """Call this precomputation function before scoring rules!

        fweights and eweights are lexical weight for each f and e word,
        normalized by their number of alignments. these weights are computed
        first at sentence level, so that each rule occurrence in this sentence
        can easily compute its lexical scores by accumulating the relevant
        weights.  However, this method computes lexical score for every
        duplicated rule occurrence throughout the corpus. The up side is that
        we don't need to keep word alignments in the output rule occurrence
        file."""
        self.fweights = None
        self.eweights = None
        self.fratios = None
        if self.fweighttable is not None:
            self.fweights = compute_weights(a, self.fweighttable)
        if self.eweighttable is not None:
            self.eweights = compute_weights(a, self.eweighttable, transpose=True)
        if self.ratiotable is not None:
            self.fratios = compute_weights(a, self.ratiotable, swap=True)

    def score_rule(self, a, r):
        funaligned = eunaligned = 0
        fweight = eweight = 1.0
        fratio = 0.0
        for i in range(len(r.f)):
            if not isvar(r.f[i]):
                if not a.faligned[r.fpos[i]]:
                    funaligned += 1
                if self.fweights is not None:
                    fweight *= self.fweights[r.fpos[i]]
                if self.fratios is not None:
                    fratio += self.fratios[r.fpos[i]]
        for i in range(len(r.e)):
            if not isvar(r.e[i]):
                if not a.ealigned[r.epos[i]]:
                    eunaligned += 1
                if self.eweights is not None:
                    eweight *= self.eweights[r.epos[i]]
        scores = []
        if self.fweights is not None:
            scores.append(fweight)
        if self.eweights is not None:
            scores.append(eweight)
        if self.fratios is not None:
            scores.append(fratio)
        return scores

def read_weightfile(f, threshold=None):
    w = {}
    progress = 0
    for line in f:
        progress += 1
        if progress % 100000 == 0:
            logger.write(".")
        (word1, word2, p) = line.split()
        p = float(p)
        if threshold is not None and p < threshold:
            continue
        if word1 == "NULL":
            word1 = None
        if word2 == "NULL":
            word2 = None
        w.setdefault(word1,{}).setdefault(word2, p)
    logger.write("done\n")
    return w

def compute_weights(a, w, transpose=False, swap=False):
    """
    transpose: compute weights for English instead of French
    swap: file format has first two columns swapped
    """
    result = []
    if not transpose:
        fwords,ewords,faligned = a.fwords, a.ewords, a.faligned
    else:
        fwords,ewords,faligned = a.ewords, a.fwords, a.ealigned
    for i in range(len(fwords)):
        total = 0.0
        n = 0
        if faligned[i]:
            for j in range(len(ewords)):
                if not transpose:
                    flag = a.aligned[i][j]
                else:
                    flag = a.aligned[j][i]
                if flag:
                    try:
                        if not swap:
                            total += w[fwords[i]][ewords[j]]
                        else:
                            total += w[ewords[j]][fwords[i]]
                    except:
                        pass
                        #logger.write("warning: couldn't look up lexical weight for (%s,%s)\n" % (fwords[i], ewords[j]))
                    n += 1
        else:
            try:
                if not swap:
                    total += w[fwords[i]][None]
                else:
                    total += w[None][fwords[i]]
            except:
                pass
                #logger.write("warning: couldn't look up null alignment for %s\n" % fwords[i])
            n += 1
        result.append(float(total)/n)
    return result

