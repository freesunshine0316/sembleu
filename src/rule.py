#!/usr/bin/env python3
import sys
import copy
import re
import gflags
FLAGS = gflags.FLAGS
#import decoding_flags
import logger
from util.hgraph.hgraph import Hgraph
from util.exceptions import DerivationException
from re_utils import delete_pattern

def nt_escape(sym):
    """Escape a string to make it a proper nonterminal symbol."""
    sym = sym.replace(',', 'COMMA')
    sym = sym.replace('[', 'LB')
    sym = sym.replace(']', 'RB')
    return sym

def get_num_edges(rule_str):
    re_edgelabel = re.compile(':[^\s\)]*')
    position = 0
    mapping = []
    num_edges = 0
    while position < len(rule_str):
        match = re_edgelabel.search(rule_str, position)
        if not match:
            break
        num_edges += 1
        position = match.end()
    return num_edges

def retrieve_edges(rule_str):
    edge_regex = re.compile(':[^\s\)]+')
    position = 0
    mapping = []
    edges = []
    while position < len(rule_str):
        match = edge_regex.search(rule_str, position)
        if not match:
            break
        edges.append(match.group(0)[1:])
        position = match.end()
    return edges

def reform_edge(s):
    re_edgelabel = re.compile(':[^\s\)]*')
    position = 0
    new_s = ''
    mapping = []
    while position < len(s):
        match = re_edgelabel.search(s, position)
        if not match:
            new_s = new_s + s[position : ]
            break
        new_s = new_s + s[position : match.start()]
        token = match.group(0)
        if token[0] == ':' and token[1] == '[' and token[-1] == ']': # :[A0,1]  or  :[S]
            sym, idx = token[2:-1].split(',')
            new_s = new_s + ':%s$%s' %(sym, idx)
            mapping.append(int(idx))
        elif token[0] == ':' and token.find('$') != -1:  # :A0$1  or  :S$
            sym, idx = token[1:].split('$')
            new_s = new_s + token
            if idx != '':
                mapping.append(int(idx))
            else:
                mapping.append(0)
        else:  # :want
            new_s = new_s + token
        position = match.end()
    #print new_s
    #print mapping
    return (new_s, mapping)


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
    return sys.intern(sym), idx

def symtostr(sym, idx):
    "Give a nonterminal an index in output."
    assert isvar(sym)
    return '[%s,%s]' % (sym[1:-1], idx)

def is_virtual(sym):
    return sym.startswith('[V_')

class Rule(object):
    """SCFG rule

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
        self.f = []  # foreign phrase
        self.e = Hgraph()
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
        #self.e2f = []
        self.grammar = None

    def init(self, lhs, f, e, e2f):
        #farity = sum(isvar(sym) for sym in f)
        #earity = sum(isvar(sym) for sym in e)
        #assert farity == earity == len(e2f)
        self.lhs = lhs
        self.f = f
        self.e = e
        self.e2f = e2f
        self.arity = sum(isvar(sym) for sym in f)

    def rank_cost(self):
        return self.cost + self.hcost

    def fromstr(self, line):
        s = line.split('|||')
        #logger.write(str(len(s)) + "\n")
        #for ele in s:
        #  logger.write(ele)
        try:
            assert len(s) >= 3, 'fewer fields than expected'
        except:
            print(line)
            print(s)
            sys.exit(-1)
        if len(s) > 3:
            lhs, f, e, probs = s
        else:
            lhs, f, e = s
            probs = ''

        #print '==============='
        #print lhs
        #print f
        #print e
        #print probs

        self.lhs, _ = symfromstr(lhs.strip())

        self.f = []
        f_var_indices = []
        f_var2idx = {}
        int_gluesymnum = 0
        for i, x in enumerate(f.split()):
            sym, idx = symfromstr(x)
            self.f.append(sym)
            if isvar(sym):
                f_var2idx[(sym, idx)] = len(f_var_indices)
                f_var_indices.append(i)
                if sym == '[A0-0]':
                    int_gluesymnum += 1

        # S ||| X0 X1 X2 ||| X1 X0 X2 --> e2f = [1,0,2]
        # GOAL ||| S ||| S --> e2f = [0]
        new_e, e_var_indices = reform_edge(e)

        #print '-------------'
        #print new_e
        #print e_var_indices

        self.e = Hgraph.from_string(new_e)

        #self.e2f = copy.copy(e_var_indices)

        if len(f_var_indices) - int_gluesymnum != len(e_var_indices):
            print('----------------')
            print(line)
            print(f_var_indices)
            print(int_gluesymnum)
            print(e_var_indices)

        assert len(f_var_indices)-int_gluesymnum == len(e_var_indices), line
        #if len(f_var_indices) != len(e_var_indices):
        #    return False

        self.feats = [float(x) for x in probs.split()]
        self.arity = sum(isvar(x) for x in self.f)

        return True

    #this method shouldn't work
    def rewrite(self, vars):
        """'vars' are lists of target side symbols. rewrite variables with lists
        of symbols in 'vars' and return the target side list of symbols after
        rewriting"""
        print("rule.rewrite shouldn't be called")
        assert False
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

        # freesunshine
        # don't need compose GOAL rule, just return its first child (and the only child)
        if self.lhs.find('GOAL') != -1: # self.lhs == '[GOAL]'
            assert len(rules) == 1
            new_rule = copy.copy(rules[0])
            return new_rule

        # useless
        # freesunshine
        # the leaf node, don't to anything too.
        #if len(rules) == 0:
        #    new_rule = copy.copy(self)
        #    if step is not None:
        #        new_rule.e = new_rule.e.clone_canonical(prefix = str(step[0]))
        #    return new_rule


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


        #logger.writeln('+++++++++++++%s' % str(self))
        #for rrr in rules:
        #    logger.writeln('           ->%s' % str(rrr))

        #if step is not None:
        #    new_e = self.e.clone_canonical(prefix = str(step[0]))
        #else:
        #    new_e = self.e.clone()

        # deal with the glue rules
        if self.lhs in ('[S]','[X]',):
            if len(rules) == 2:
                hgs = [x.e for x in rules]
                comb_hg = Hgraph.combine_multiple(hgs)
            else:
                assert(len(rules) == 1)
                comb_hg = rules[0].e.clone()
            new_rule = Rule()
            new_rule.init(self.lhs, new_f, comb_hg, [])
            new_rule.we2f = []
            return new_rule

        new_e = self.e.clone()
        for i,sub_rule in enumerate(rules):

            # added by freesunshine, to ignore the glue symbol, when compose the rules
            # For example, ---> [A2-10] ||| [A1-1] [A0-0] ||| (. :A1-1 ), there is no
            #                   +++ [A1-1] ||| ||| the graph fragment
            #                   +++ [A0-0] ||| ||| .
            # Actually we shouldn't compose [A2-10] with [A0-0]
            if sub_rule.lhs == '[A0-0]':
                continue

            sub_e = sub_rule.e.clone_canonical(prefix = str(i))
            p,r,c = new_e.find_nt_edge(sub_rule.lhs[1:-1],str(i))
            fragment = Hgraph.from_triples([(p,r,c)],self.e.node_to_concepts)
            #print >>sys.stderr, '=======before replacement======='
            #print >>sys.stderr, new_e
            #print >>sys.stderr, fragment
            #print >>sys.stderr, sub_e
            #print >>sys.stderr, '--------------------------------'
            try:
                new_e = new_e.replace_fragment(fragment,sub_e)
            except AssertionError as e:
                #print fragment
                #print sub_e
                raise DerivationException("Incompatible hyperedge type for nonterminal %s$%s." % (sub_rule.lhs[1:-1],str(i)))
            #print >>sys.stderr, '=======after replacement======='
            #print >>sys.stderr, new_e
            #print >>sys.stderr, '-------------------------------'
            #sys.exit(0)

        new_rule = Rule()
        new_rule.init(self.lhs, new_f, new_e, [])
        new_rule.we2f = []
        #print >>sys.stderr, 'Ori:', self.e
        #print >>sys.stderr, 'New:', new_e
        return new_rule

    def fwi_aligned_to_ewi(self, ewi):
        if hasattr(self, 'we2f'):
            return self.we2f[ewi]
        else:
            return None

    def dumped_format(self):
        f_strs = []
        i = 1
        f_lexed = False
        self.feats = [1.0]
        for sym in self.f:
            if isvar(sym):
                f_strs.append(symtostr(sym, i))
                i += 1
            else:
                f_strs.append(str(sym))
                f_lexed = True
        if f_lexed:
            self.feats.append(0.0)
        else:
            self.feats.append(1.0)
        e_str = str(self.e)
        match = re.search('(:[^A\s\n]+)|(:A[^0-9]+)|(:A[0-9]+[\s]+)', e_str)
        if match:
            self.feats.append(0.0)
        else:
            self.feats.append(1.0)
        result= '%s ||| %s ||| %s ||| %s' % \
                (self.lhs,
                 ' '.join(f_strs),
                 e_str,
        #         ' '.join(e_strs),
                 ' '.join([str(f) for f in self.feats]))
        result = delete_pattern(result, '~e\.[0-9]+(,[0-9]+)*')
        return result

    def __str__(self):
        # f side always straight
        f_strs = []
        i = 1
        #f_lexed = False
        #self.feats = [1.0]
        for sym in self.f:
            if isvar(sym):
                f_strs.append(symtostr(sym, i))
                i += 1
            else:
                f_strs.append(str(sym))
                #f_lexed = True
        #if f_lexed:
        #    self.feats.append(0.0)
        #else:
        #    self.feats.append(1.0)
        e_str = str(self.e)
        #match = re.search('(:[^A\s\n]+)|(:A[^0-9]+)|(:A[0-9]+[\s]+)', e_str)
        #if match:
        #    self.feats.append(0.0)
        #else:
        #    self.feats.append(1.0)
        #print 'e_str: %s' % e_str
        # e side permuted
        #e_strs = []
        #i = 0
        #for sym in self.e:
        #    if isvar(sym):
        #        e_strs.append(symtostr(sym, self.e2f[i] + 1))
        #        i += 1
        #    else:
        #        e_strs.append(str(sym))
        result= '%s ||| %s ||| %s%s' % \
                (self.lhs,
                 ' '.join(f_strs),
                 e_str,
        #         ' '.join(e_strs),
               ' ||| ' +  ' '.join([str(f) for f in self.feats]) if len(self.feats) > 0 else '')
        return result

    def __lt__(self, other):
        return self.rank_cost() < other.rank_cost()

    def key(self):
        return (self.lhs, tuple(self.f), self.e)

    def __eq__(self, other):
        return self.key() == other.key()

    def __hash__(self):
        return hash(self.key())

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
