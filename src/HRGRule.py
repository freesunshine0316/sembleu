#!/usr/bin/env python3
import sys
from util.hgraph.hgraph import Hgraph
import gflags
FLAGS = gflags.FLAGS
#import decoding_flags
import logger

def nt_escape(sym):
    """Escape a string to make it a proper nonterminal symbol."""
    sym = sym.replace(',', 'COMMA')
    sym = sym.replace('[', 'LB')
    sym = sym.replace(']', 'RB')
    return sym

def isvar(s):
    return s.startswith('[') and s.endswith(']')

def nocat(s):
    "Return a var with category removed."
    tmp = s[1:-1].split('-')
    if len(tmp) == 2:
        return '[%s]' % tmp[0]
    else:
        return s

def symfromstr(s):
    "Read terminal or nonterminal from a string."
    idx = None
    if isvar(s):
        s = s[1:-1].split(',')
        if len(s) == 2:
            idx = int(s[1])
        nt = s[0].strip()
        sym = '[%s]' % nt
    else:
        sym = s
    # symbols are interned to save memory
    return intern(sym), idx

def symtostr(sym, idx):
    "Give a nonterminal an index in output."
    assert isvar(sym)
    return '[%s,%s]' % (sym[1:-1], idx)

def is_virtual(sym):
    return sym.startswith('[V_')

class SHRGRule(object):
    """SHRG rule

    >>> r = Rule()
    >>> r.fromstr('[X] ||| xx [C] [B] [A,2] [A,1] ||| aa [A,1] cc [A,2] dd [B] [C] ||| 1.0 0 0 0')
    >>> r.we2f = [None, 4, None, None, 0, 2, 1]
    >>> str(r)
    '[X] ||| xx [C,1] [B,2] [A,3] [A,4] ||| aa [A,4] cc [A,3] dd [B,2] [C,1] ||| 1.0 0.0 0.0 0.0'
    >>> r.arity
    4
    >>> r.scope()
    4
    >>> r.rewrite([['A1', 'A1'], ['A2'], ['B', 'B'], ['C']])
    ['aa', 'C', 'cc', 'B', 'B', 'dd', 'A2', 'A1', 'A1']
    >>> r1 = Rule()
    >>> r1.fromstr('[A] ||| [D] subxx1 [G] subxx2 ||| sub1 [D] sub2 [G] ||| 1.0 0 0 0')
    >>> r1.we2f = [1, 0, 3, 2]
    >>> r2 = Rule()
    >>> r2.fromstr('[B] ||| [F] [E] ||| [E] [F] ||| 1.0 0 0 0')
    >>> r2.we2f = [1, None]
    >>> new_r = r.compose([None, r2, r1, r1])
    >>> print(new_r)
    [X] ||| xx [C,1] [F,2] [E,3] [D,4] subxx1 [G,5] subxx2 [D,6] subxx1 [G,7] subxx2 ||| aa sub1 [D,6] sub2 [G,7] cc sub1 [D,4] sub2 [G,5] dd [E,3] [F,2] [C,1] |||
    >>> print(new_r.we2f)
    [None, 9, 8, 11, 10, None, None, None, None, None, 0, 3, None, 1]
    """

    symbol_table = {}

    def __init__(self):
        self.lhs = None  # lhs nonterminal
        self.s = []  # string side
        self.g = Hgraph()  # graph side
        # this is weighted sum of self.fcosts
        # precomputed to be used in decoding
        self.cost = 0
        self.hcost = 0
        # list of feature values read from input
        self.feats = []
        # list of feature values computed by stateless feature functions
        # these are real costs used in decoding.
        # this is different from feature fields read from input because
        # feature funtions may not pick all the features in input file,
        # may use log probabilities, etc.
        self.fcosts = []
        self.cost = 0
        # e2f[i] = j means the i'th nonterminal on the e side is linked to
        # the j'th nonterminal on the f side. nonterminal permutation
        # information is saved only here, the idx assigned to nonterminals
        # are only for input/output purposes.
        #self.e2f = [] #The string to graph nonterminal mapping is kept in the edge
        self.grammar = None

    def init(self, lhs, f, e, e2f):
        farity = sum(isvar(sym) for sym in f)
        earity = sum(isvar(sym) for sym in e)
        assert farity == earity == len(e2f)
        self.lhs = lhs
        self.f = f
        self.e = e
        self.e2f = e2f
        self.arity = farity

    def rank_cost(self):
        return self.cost + self.hcost

    def fromstr(self, line):
        s = line.split('|||')
        #logger.write(str(len(s)) + "\n")
        #for ele in s:
        #  logger.write(ele)
        assert len(s) == 4, 'fewer fields than expected'
        lhs, f, e, probs = s
        self.lhs, _ = symfromstr(lhs.strip())

        self.f = []
        f_var_indices = []
        f_var2idx = {}
        for i, x in enumerate(f.split()):
            sym, idx = symfromstr(x)
            self.f.append(sym)
            if isvar(sym):
                f_var2idx[(sym, idx)] = len(f_var_indices)
                f_var_indices.append(i)

        self.e = []
        e_var_indices = []
        self.e2f = []  # maps e var indices to f var indices
        for i, x in enumerate(e.split()):
            sym, idx = symfromstr(x)
            self.e.append(sym)
            if isvar(sym):
                self.e2f.append(f_var2idx[(sym, idx)])
                e_var_indices.append(i)

        assert len(f_var_indices) == len(e_var_indices), line

        self.feats = [float(x) for x in probs.split()]
        self.arity = sum(isvar(x) for x in self.e)

    def rewrite(self, vars):
        """'vars' are lists of target side symbols. rewrite variables with lists
        of symbols in 'vars' and return the target side list of symbols after
        rewriting"""
        assert len(vars) == self.arity
        result = []
        e_var_idx = 0
        for sym in self.e:
            if isvar(sym):
                f_var_idx = self.e2f[e_var_idx]
                result += vars[f_var_idx]
                e_var_idx += 1
            else:
                result.append(sym)
        return result

    def compose(self, rules):
        """rules is list of rules to be composed with self. A None in the list
        means the nonterminal is not rewritten. Returns a composed rule. rules
        are given in the order of f side nonterminals."""
        assert len(rules) == self.arity
        fi = 0
        if not FLAGS.nt_mismatch:
            for sym in self.f:
                if isvar(sym) and rules[fi] is not None:
                    assert rules[fi].lhs == sym, \
                            'cannot rewrite nonterminal %s of %s with %s' % \
                            (fi, self, rules[fi])
                    fi += 1
        # a map: fi_sub_fi2new_fi[fi][sub_fi] = new_fi
        # where fi is f var idx in top level rule (self), sub_fi is f var idx
        # in a sub rule. new_fi is the f var idx in the composed rule
        #
        # a speical case is that when a sub rule is None:
        # fi_sub_fi2new_fi[fi] = new_fi
        fi_sub_fi2new_fi = []
        new_f = []
        new_fi = 0  # f var idx in composed rule
        fi = 0  # f var idx in self

        # a similar map, applied to all symbols, both terminals and nonterminals
        fwi_sub_fwi2new_fwi = []
        new_fwi = 0  # f symbol idx in composed rule

        for fwi, sym in enumerate(self.f):
            if isvar(sym):
                r = rules[fi]
                if r is None:
                    new_f.append(sym)

                    fi_sub_fi2new_fi.append(new_fi)
                    fwi_sub_fwi2new_fwi.append(new_fwi)

                    new_fi += 1
                    new_fwi += 1
                else:
                    new_f.extend(r.f)

                    sub_fi2new_fi = []
                    for i in range(r.arity):
                        sub_fi2new_fi.append(new_fi + i)
                    fi_sub_fi2new_fi.append(sub_fi2new_fi)

                    sub_fwi2new_fwi = []
                    for i in range(len(r.f)):
                        sub_fwi2new_fwi.append(new_fwi + i)
                    fwi_sub_fwi2new_fwi.append(sub_fwi2new_fwi)

                    new_fi += r.arity
                    new_fwi += len(r.f)
                fi += 1
            else:
                new_f.append(sym)

                fwi_sub_fwi2new_fwi.append(new_fwi)
                new_fwi += 1

        new_e2f = []  # new var alignment
        new_we2f = []  # new symbol alignment
        new_e = []
        ei = 0  # e var idx in self
        new_ewi = 0  # e symbol idx in composed rule
        for ewi, sym in enumerate(self.e):
            fwi = self.fwi_aligned_to_ewi(ewi)
            if fwi is None:
                new_fwi = None
            else:
                new_fwi = fwi_sub_fwi2new_fwi[fwi]
            if isvar(sym):
                fi = self.e2f[ei]
                r = rules[fi]
                if r is None:
                    new_e.append(sym)

                    new_e2f.append(fi_sub_fi2new_fi[fi])

                    new_we2f.append(new_fwi)
                else:
                    new_e.extend(r.e)

                    for sub_ei in range(r.arity):
                        sub_fi = r.e2f[sub_ei]
                        new_e2f.append(fi_sub_fi2new_fi[fi][sub_fi])

                    for sub_ewi in range(len(r.e)):
                        sub_fwi = r.fwi_aligned_to_ewi(sub_ewi)
                        if sub_fwi is not None and fwi is not None:
                            new_we2f.append(fwi_sub_fwi2new_fwi[fwi][sub_fwi])
                        else:
                            new_we2f.append(None)
                ei += 1
            else:
                new_e.append(sym)

                new_we2f.append(new_fwi)

        new_rule = Rule()
        new_rule.init(self.lhs, new_f, new_e, new_e2f)
        new_rule.we2f = new_we2f
        return new_rule

    def fwi_aligned_to_ewi(self, ewi):
        if hasattr(self, 'we2f'):
            return self.we2f[ewi]
        else:
            return None

    def __str__(self):
        # f side always straight
        #assert False, 'This method is not well implemented, avoid here'
        return 'graph side:%s\nstring side:%s\n' % (str(self.g), ' '.join(self.s))

    def __lt__(self, other):
        return self.rank_cost() < other.rank_cost()

    def key(self):
        return (self.lhs, tuple(self.s), self.g)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        hash_value = 0
        for item in self.key():
            hash_value += hash(item)
        return hash_value
        #return hash(self.key())

    def scope(self):
        v = [isvar(sym) for sym in self.f]
        return sum(a and b for a, b in zip([True] + v, v + [True]))

    def lexical(self):
        return True if len(self.e2f) == 0 else False

    def nonlexical(self):
        return True if all(isvar(sym) for sym in self.f) else False

    def align_special_symbols(self):
        """A hack. If both source and target side contain only one special
        terminal beginning with '$', align them.
        Return True if anything there are special symbol alignments found.
        """
        self.we2f = [None] * len(self.e)
        result = False

        fi2fwi = []  # map from var idx to word idx
        for fwi, fw in enumerate(self.f):
            if isvar(fw):
                fi2fwi.append(fwi)

        ei = 0
        for ewi, ew in enumerate(self.e):
            if ew in ('$number', '$date'):
                for fwi, fw in enumerate(self.f):
                    if fw == ew:
                        self.we2f[ewi] = fwi
                        result = True
            elif isvar(ew):
                self.we2f[ewi] = fi2fwi[self.e2f[ei]]
                ei += 1

        return result

    def fnts(self):
        "Iterate over f side nts."
        return [x for x in self.f if isvar(x)]
        #for word in self.f:
        #    if isvar(word):
        #        yield word

if __name__ == '__main__':
    import doctest
    doctest.testmod()
    #from mymonitor import memory, human
    #import sys
    #rules = []
    #for i, line in enumerate(sys.stdin):
    #    if i % 10000 == 0:
    #        print('mem: %s' % human(memory()))
    #    rule = Rule()
    #    rule.fromstr(line)
    #    rules.append(rule)
