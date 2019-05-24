from hypergraph import *
from levels import mark_level
from util.hgraph.hgraph import Hgraph
from util.cfg import NonterminalLabel
from HRGRule import SHRGRule
from rule import Rule
#from hrg_sampler import NPSampler
import random
import sys
from math import log, exp, factorial, lgamma
import logprob
from lexical_weighter import LexicalWeighter
from common import INF, ZERO
from levels import mark_level
#from monitor import memory, resident
from collections import deque
from operator import itemgetter
from filter_stop_words import filter_vars
#from amr_fragment import check_consist
def timed(l):
    for i, x in enumerate(l, 1):
        yield x

def lncr(n, r):
    "log choose r from n"
    return lgamma(n+1) - lgamma(r+1) - lgamma(n-r+1)

def rule_size(rule):
    return  len(rule.s) + rule.g.num_edges# - 2 * rule.arity + rule.scope()

def geometric(rule):
    p = 0.99

    l = rule_size(rule)
    if l == 0:
        return 1.0
    else:
        return (1-p)**(l-1) * p

def poisson(rule):
    mean = 2.0
    l = rule_size(rule)

    denominator = log(factorial(l))
    numerator = l * log(mean) - mean

    result = exp(numerator - denominator)

    return result if result > ZERO else ZERO

def weibull(rule):
    shape = 2.9
    scale = 2.3

    l = rule_size(rule)

    result = ((shape/scale)
            * (pow(float(l)/scale, shape-1))
            * exp(-pow(float(l)/scale, shape)))

    return result if result > ZERO else ZERO

def uniform(rule):
    return 1.0

base = poisson
rule_size_prob = poisson
# emulate a Dirichlet distribution with three events
# used for the ab_test
def abtest_base(base):
    return 1.0/3

def uniform_base(rule):
    l = rule_size(rule)

    result = pow(2.0, -l)
    return result if result > ZERO else ZERO

def choose_split(n, choices, corrections, sampler):
    assert corrections is not None
    #sample for n times, get the corresponding counts for each ni
    n_edges = len(choices)
    counts = [0 for x in xrange(n_edges)]
    for i in xrange(n):
        result = discrete([sampler.choice_posterior(c)*f for c, f in zip(choices, corrections)])
        counts[result] += 1

    return counts

class Sampler(object):
    def count(self, x):
        pass

    def discount(self, x):
        pass

    def posterior(self, x):
        return 1.0

    def choice_posterior(self, c):
        result = 1.0
        for x in c:
            result *= self.posterior(x)
            if result < ZERO:
                return ZERO
        return result

class NPSampler(Sampler):
    "NP for non-paramatric"
    def __init__(self):
        self.counts = {}
        self.rule_size_counts = {}  # key=rule size, value=counts
        self.rule_size_tables = {}  # key=rule size, value=estimated tables
        self.n = 0

    def update_rule_size_tables(self):
        self.rule_size_tables = {}

        # should be fixed for python3
        for rule, counts in self.counts.iteritems():
            l = rule_size(rule)
            estimated_number_of_tables = pow(counts, FLAGS.discount)
            self.rule_size_tables[l] =  (self.rule_size_tables.get(l, 0.0)
                                            + estimated_number_of_tables)

    '''
    This is used to calculate the posterior prob of one rule
    Before this procedure, should discount the rule counts
    '''
    def pitman_yor_posterior_rule_size(self, x):
        alpha = FLAGS.alpha
        if FLAGS.variable_alpha == True:
            alpha = FLAGS.alpha * rule_size_prob(x)

        l = rule_size(x)
        n_r = self.counts.get(x, 0.0)
        n_l = self.rule_size_counts.get(l, 0.0)
        T_r = pow(n_r, FLAGS.discount)
        T_l = self.rule_size_tables.get(l, 0.0)

        return (((n_r - T_r*FLAGS.discount + (T_l*FLAGS.discount + alpha)*base(x))
                * (rule_size_prob(x)))
                / (n_l + alpha))

    # Used for generating likelihood graph
    # Single Dirichlet process, no rule size
    def simple_dirichlet_posterior(self, x):
        n_r = self.counts.get(x, 0.0)
        return ((n_r + FLAGS.alpha*base(x))
                / (self.n + FLAGS.alpha))

    def simple_dirichlet_posterior_for_choice(self, c):
        result = 1.0

        alpha = FLAGS.alpha
        for x in c:
            alpha = FLAGS.alpha
            n_r = self.counts.get(x, 0.0)

            result *= ((n_r + alpha*base(x))
                        / (self.n + alpha))

            self.count(x)

        # To make it absolutely correct
        for x in c:
            self.discount(x)

        return result

    '''
    xpeng: add one to the rule count of x and the total rule count
    if Pitman-Yor, add one to the rule size count of rule_size(x)
    '''
    def count(self, x):
        #print('count %s' % x)
        self.counts[x] = self.counts.get(x, 0) + 1
        self.n += 1

        if FLAGS.model == 'PY':
            l = rule_size(x)
            self.rule_size_counts[l] = self.rule_size_counts.get(l, 0) + 1

    def top_stats(self):
        stats = sorted(self.counts.items(), key=itemgetter(1), reverse=True)
        return stats[0:10]

    '''
    xpeng: extract rule count of x and total rule count by 1
    if Pitman-Yor, then extract rule size count of rule_size(x) by 1
    there should be deletion if count is 0
    '''
    def discount(self, x):
        #print('discount %s' % x)
        if FLAGS.model == 'PY':
            l = rule_size(x)
            try:
                c = self.rule_size_counts[l]
                if c == 1:
                    del self.rule_size_counts[l]
                else:
                    self.rule_size_counts[l] = c - 1
            except KeyError:
                logger.writeln('Warning: rule size %d not seen before in discounting' % l)

        try:
            c = self.counts[x]
            if c == 1:
                del self.counts[x]
            else:
                self.counts[x] = c - 1
            self.n -= 1
        except KeyError:
            logger.writeln('Warning: %s not seen before in discounting' % x)

    def add(self, other):
        self.n += other.n
        for rule in other.counts.keys():
            self.counts[rule] = self.counts.get(rule, 0)+ other.counts[rule]
            assert self.counts[rule] >= 0, 'Meet negative rules'
            if self.counts[rule] == 0:
                del self.counts[rule]

        if FLAGS.model == 'PY':
            for rule_size in other.rule_size_counts.keys():
                self.rule_size_counts[rule_size] = self.rule_size_counts.get(rule_size, 0) + other.rule_size_counts[rule_size]

    def substract(self, other):
        self.n -= other.n
        #assert self.n > 0, 'Number of updated rules should be positive'
        for rule in other.counts.keys():
            self.counts[rule] = self.counts.get(rule, 0)- other.counts[rule]
            #assert self.counts[rule] >= 0, 'The count of a rule turned negative'
            if self.counts[rule] == 0:
                del self.counts[rule]

        if FLAGS.model == 'PY':
            for rule_size in other.rule_size_counts.keys():
                self.rule_size_counts[rule_size] = self.rule_size_counts.get(rule_size, 0) - other.rule_size_counts[rule_size]
                #assert self.rule_size_counts[rule_size] >= 0, 'The count of a rule size turned negative'
                if self.rule_size_counts[rule_size] == 0:
                    del self.rule_size_counts[rule_size]

    # Now this is really wrong...
    def rule_size_likelihood(self):
        result = 0.0
        for rule, count in self.counts.iteritems():
            l = rule_size(rule)
            try:
                if l not in self.rule_size_counts:
                    logger.writeln('meet unseen rule count %d' % l)
                f_n_l = float(self.rule_size_counts[l])
            except KeyError:
                logger.writeln(str(l))
                logger.writeln(self.rule_size_counts[l])
                assert False, 'Get out here'

            l_prob = rule_size_prob(rule)
            result += count * logprob.elog(float(count) * l_prob / f_n_l)
        return result

    # Only works for dirichlet process, no rule size
    def dp_likelihood(self):
        n = 0.0
        counts = {}
        result = 0.0
        alpha = FLAGS.alpha
        for r, count in self.counts.iteritems():
            for _ in range(count):
                n_r = float(counts.get(r, 0.0))
                result += logprob.elog((n_r+alpha*base(r))/(n+alpha))
                counts[r] = counts.get(r, 0.0) + 1.0
                n += 1.0
        return result

    def nsamples(self):
        return self.n

    def ntypes(self):
        return len(self.counts)

class NTSampler(Sampler):
    "NT for nonterminal"
    def __init__(self):
        self.samplers = {}

    def count(self, x):
        self.samplers.setdefault(x.lhs, NPSampler()).count(x)

    def discount(self, x):
        self.samplers.setdefault(x.lhs, NPSampler()).discount(x)

    def posterior(self, x):
        return self.samplers.setdefault(x.lhs, NPSampler()).posterior(x)

    def update_rule_size_tables(self):
        for sampler in self.samplers.itervalues():
            sampler.update_rule_size_tables()

    def nsamples(self):
        return sum(s.nsamples() for s in self.samplers.itervalues())

    def ntypes(self):
        return sum(s.ntypes() for s in self.samplers.itervalues())

    def likelihood(self):
        return sum(s.likelihood() for s in self.samplers.itervalues())

def choice(choices, i, SAMPLER, correction=None):
    "Uses a global sampler to choose among a few options."
    # when we use double count sample_cut and sample_edge
    # call Sampler.double_count and Sampler.double_discount
    # by themselves
    if not FLAGS.double_count:
        for x in choices[i]:
            SAMPLER.discount(x)

    if FLAGS.correct_edge_sampling or correction is None:
        weights = [SAMPLER.choice_posterior(c) for c in choices]
        if FLAGS.anneal and not FLAGS.cease_update:
            weights = [x ** FLAGS.annealing_factor for x in weights]
        result = discrete(weights)
    else:
        assert len(choices) == len(correction), \
               'len(choices)=%s, len(correction)=%s' % (len(choices),
                                                        len(correction))
        weights = [SAMPLER.choice_posterior(c)*f for c, f in zip(choices, correction)]
        if FLAGS.anneal and not FLAGS.cease_update:
            weights = [x ** FLAGS.annealing_factor for x in weights]
        result = discrete(weights)

    if not FLAGS.double_count:
        for x in choices[result]:
            SAMPLER.count(x)

    return result

def discrete(l):
    s = sum(l)
    if s == 0.0:
        length =  len(l)
        l = [1/length] * length
    else:
        l = [x/s for x in l]
    s = 0.0
    r = random.random()
    for i, x in enumerate(l):
        s += x
        if r <= s:
            return i
'''
xpeng: the Sample class is used to record one sample of sentences(hypergraph-structured)
'''
class Sample():
    '''
    xpeng: initialization
    1.init the hypergraph and alignment attribute
    2.for each node, init the level, cut, chosen-edge and parent info
    '''
    def __init__(self, hg, sent_num, SAMPLER=None):
        self.hg = hg  # hypergraph
        mark_level(self.hg) # xpeng:calculate the level attribute for each node in the hypergraph, in topological order
        '''
        This is to iterate all the nodes, which does not specify one tree
        '''
        self.SAMPLER = SAMPLER
        self.sent_no = sent_num
        for node in self.hg.nodes:
            node.cut = 1 # xpeng: each node is initiated as a cut-site

            if node.nosample:
                node.cut = 0  #This is to constrain the number of nonterminals
            if len(node.incoming) == 0:
                continue

            node.edge = random.randrange(len(node.incoming)) # xpeng: randomly choose incoming edge for each node
            node.nt = 0
            node.pnt = None  # parent nt (before splitting)
            node.sampled = False

        set_parents_under(self.hg.root)

    def set_sampler(self, SAMPLER):
        self.SAMPLER = SAMPLER

    '''
    xpeng: make a composed rule with the node and all its cut-node descendants
    rule composition remains to be implemented for HRG rule
    '''
    def make_composed_rule(self, node, first_rule=False):
        assert node.cut == 1, 'Non-cut node can not be root of a subtree, %s' % str(node.frag)
        mychildren = children(node)
        shrg_rule = SHRGRule()
        var_mapping = {}

        n_ext = len(node.frag.ext_set)

        suffix = self.get_node_suffix(node, list(node.frag.ext_set))

        if node.frag.root in node.frag.ext_set or first_rule:
            shrg_rule.lhs = intern('[A%d-%s]' % (n_ext, suffix))
        else:
            shrg_rule.lhs = intern('[A%d-%s]' % (n_ext+1, suffix))

        #shrg_rule.lhs = 'A%d' % len(node.frag.ext_set) #Newly corrected
        (shrg_rule.s, nonterm_index) = self.initialize_string_side(node, mychildren)
        self.initialize_graph_side(shrg_rule.g, node, mychildren, nonterm_index)
        return shrg_rule

    def initialize_string_side(self, node, children):
        if children is None: #The current node is already the leaf
            if node.frag.start == -1:
                return ([], {})
            return (node.frag.graph.sent[node.frag.start:node.frag.end], {})

        s_side = []
        self.verify_order(children)
        start = -1
        graph = node.frag.graph
        nonterm_index = {}
        index = 0
        end = node.frag.start

        for child_node in children:
            curr_start = child_node.frag.start
            if curr_start != -1:
                if end != -1: #There are already some aligned fragments before
                    s_side.extend(graph.sent[end:curr_start])

            if child_node.cut == 1: #Current node is a cut, so rewrite it with a nonterminal
                s_side.append('A')
                nonterm_index[child_node] = index
                index += 1
            else:
                s_side.extend(graph.sent[child_node.frag.start:child_node.frag.end])

            if curr_start != -1:
                start = curr_start
                end = child_node.frag.end
        assert node.frag.end == end, 'The composition of fragments results in inconsitent positions'
        return (s_side, nonterm_index)

    def initialize_new_graph_side(self, hgraph, node, children, nonterm_index, ext_mapping, att_list_mapping):
        var_mapping = {}
        if children is None:
            #hgraph.roots.append(node.frag.root)
            visited_index = set()
            node.frag.init_new_hgraph(hgraph, node.frag.root, var_mapping, visited_index, None, None, None, ext_mapping, att_list_mapping) #To be fixed
            return

        assert check_node_consist(node, children), 'Unexpected parent %s with span %d-%d' % (str(node.frag), node.frag.start, node.frag.end)

        #if len(hgraph.roots) == 0:
        #    hgraph.roots.append(node.frag.root)
        unvisited_nodes = set(children)
        self.new_graph_under_node(node.frag.root, hgraph, var_mapping, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping, True)
        try:
            assert len(unvisited_nodes) == 0, 'There are still unvisited nodes'
        except AssertionError:
            logger.writeln('Unvisited nodes happen in sent %d' % self.sent_no)
            for n in nodes_in_fragment_under(node):
                print n.frag.graph_side({})

    def initialize_graph_side(self, hgraph, node, children, nonterm_index):
        #hgraph = Hgraph()
        var_mapping = {}
        if children is None:
            #hgraph.roots.append(node.frag.root)
            visited_index = set()
            node.frag.init_hgraph(hgraph, node.frag.root, var_mapping, visited_index)
            return

        assert check_node_consist(node, children), 'Unexpected parent %s with span %d-%d' % (str(node.frag), node.frag.start, node.frag.end)

        #if len(hgraph.roots) == 0:
        #    hgraph.roots.append(node.frag.root)
        unvisited_nodes = set(children)
        self.graph_under_node(node.frag.root, hgraph, var_mapping, nonterm_index, unvisited_nodes)
        try:
            assert len(unvisited_nodes) == 0, 'There are still unvisited nodes'
        except AssertionError:
            logger.writeln('Unvisited nodes happen in sent %d' % self.sent_no)
            for n in nodes_in_fragment_under(node):
                print n.frag.graph_side({})

    def graph_under_node(self, curr_root, hgraph, var_mapping, nonterm_index, unvisited_nodes):
        root_ident = None
        if curr_root not in var_mapping:
            root_ident = 'N%d' % var_mapping.setdefault(curr_root, len(var_mapping))
            ignoreme = hgraph[root_ident]
            if len(hgraph.roots) == 0:
                hgraph.roots.append(root_ident)

        root_ident = 'N%d' % var_mapping[curr_root]
        assert root_ident in hgraph, 'An index identity not registered found'

        #Find a set of candidate nodes having the current index as their roots
        candidates = set()
        for node in unvisited_nodes.copy():
            if node.frag.root == curr_root:
                candidates.add(node)
                unvisited_nodes.remove(node)

        if len(candidates) == 0:
            return

        for cand_node in candidates:
            if cand_node.cut == 1: #means the current node is a nonterminal edge
                assert cand_node in nonterm_index.keys(), 'There are unappeared nonterminal on the HRG side'
                curr_edge_id = '[A%d,%d]' % (len(cand_node.frag.ext_set), nonterm_index[cand_node])

                child_ids = []
                hgraph.num_edges += 1
                for curr_ext_index in cand_node.frag.ext_set:
                    if curr_ext_index == curr_root:
                        continue
                    if curr_ext_index not in var_mapping:
                        curr_ext_ident = 'N%d' % var_mapping.setdefault(curr_ext_index, len(var_mapping))
                    curr_ext_ident = 'N%d' % var_mapping[curr_ext_index]
                    ignoreme = hgraph[curr_ext_ident]
                    child_ids.append(curr_ext_ident)

                    self.graph_under_node(curr_ext_index, hgraph, var_mapping, nonterm_index, unvisited_nodes)
                hgraph._add_triple(root_ident, curr_edge_id, tuple(child_ids))

            else:  #which means the current node should be rewritten as terminal fragments having the current node as root

                assert cand_node not in nonterm_index.keys(), 'A terminal node found on the string side'
                visited_index = set()
                cand_node.frag.init_hgraph(hgraph, curr_root, var_mapping, visited_index, self, nonterm_index, unvisited_nodes) #This representation does not have the root in it


    def new_graph_under_node(self, curr_root, hgraph, var_mapping, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping, is_root=False):
        root_ident = None
        if curr_root not in var_mapping:
            root_ident = '_%d' % var_mapping.setdefault(curr_root, len(var_mapping))
            ignoreme = hgraph[root_ident]
            hgraph.node_to_concepts[root_ident] = None #No node label in our grammar
            if is_root:
                hgraph.roots.append(root_ident)

        root_ident = '_%d' % var_mapping[curr_root]
        if curr_root in ext_mapping and not is_root:
            if root_ident in hgraph.external_nodes and hgraph.external_nodes[root_ident] != ext_mapping[curr_root]:
                assert False, 'external nodes mapping inconsitency error!'
            hgraph.external_nodes[root_ident] = ext_mapping[curr_root]
            hgraph.rev_external_nodes[ext_mapping[curr_root]] = root_ident

        assert root_ident in hgraph, 'An index identity not registered found'

        #Find a set of candidate nodes having the current index as their roots
        candidates = set()
        for node in unvisited_nodes.copy():
            if node.frag.root == curr_root:
                candidates.add(node)
                unvisited_nodes.remove(node)

        if len(candidates) == 0:
            return

        for cand_node in candidates:
            if cand_node.cut == 1: #means the current node is a nonterminal edge
                assert cand_node in nonterm_index.keys(), 'There are unappeared nonterminal on the HRG side'
                #curr_edge_lab = 'A%d' % len(cand_node.frag.ext_set)
                #curr_edge_id = '[A%d,%d]' % (len(cand_node.frag.ext_set), nonterm_index[cand_node])

                child_ids = []
                hgraph.num_edges += 1
                curr_att_list = []
                if curr_root in cand_node.frag.ext_set:
                    curr_att_list.append(curr_root)

                for curr_ext_index in cand_node.frag.ext_set:
                    if curr_ext_index == curr_root:
                        continue
                    curr_att_list.append(curr_ext_index)
                    if curr_ext_index not in var_mapping:
                        curr_ext_ident = '_%d' % var_mapping.setdefault(curr_ext_index, len(var_mapping))
                    curr_ext_ident = '_%d' % var_mapping[curr_ext_index]
                    ignoreme = hgraph[curr_ext_ident]
                    child_ids.append(curr_ext_ident)

                    self.new_graph_under_node(curr_ext_index, hgraph, var_mapping, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping)
                assert cand_node not in att_list_mapping, 'Repeat key found'
                att_list_mapping[cand_node] = curr_att_list

                suffix = self.get_node_suffix(cand_node, curr_att_list)
                #if FLAGS.href:
                #    suffix = self.get_node_externals(cand_node, curr_att_list)

                if curr_root in cand_node.frag.ext_set:
                    curr_edge_lab = 'A%d-%s' % (len(cand_node.frag.ext_set), suffix)
                else:
                    curr_edge_lab = 'A%d-%s' % (len(cand_node.frag.ext_set) + 1, suffix) #Newly changed
                curr_edge_id = nonterm_index[cand_node]
                new_edge = NonterminalLabel(curr_edge_lab, curr_edge_id)

                hgraph._add_triple(root_ident, new_edge, tuple(child_ids))

            else:  #which means the current node should be rewritten as terminal fragments having the current node as root

                assert cand_node not in nonterm_index.keys(), 'A terminal node found on the string side'
                visited_index = set()
                cand_node.frag.init_new_hgraph(hgraph, curr_root, var_mapping, visited_index, self, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping) #This representation does not have the root in it

    '''
    return all the composed rules in the sub-tree rooted at the node
    return value: (cut node, rule)
    '''
    def composed_rules_under(self, node):
        rule = self.make_composed_rule(node)
        yield node, rule
        mychildren = children(node)
        if mychildren is not None:
            for child in mychildren:
                if child.cut == 1:
                    for n, rule in self.composed_rules_under(child):
                        yield n, rule

    #Extract the derivation for one string-graph pair
    def extract_derivation(self):
        derivation_rules = []
        root_node = self.hg.root
        reset = False
        if root_node.frag.graph.sent != root_node.frag.str_list():
            reset = True
            logger.writeln('The original sentence is: %s' % ' '.join(root_node.frag.graph.sent))
        #logger.writeln('sentence is: %s' % root_node.frag.str_side())
        assert len(root_node.frag.ext_set) == 0, 'The whole amr graph has external nodes'

        stack = [(root_node, 0, [])]
        result = ''
        first = True
        while len(stack) > 0:
            curr_root, curr_depth, curr_ext_list = stack.pop()
            leafs = children(curr_root)

            (hrg_rule, att_list_mapping) = self.extract_one_rule(curr_root, leafs, curr_ext_list, first) #This should return a ordered list of attachment nodes for each leaf
            if hrg_rule:
                derivation_rules.append(hrg_rule)

            if curr_root == root_node and reset:
                temp = (curr_root.frag.start, curr_root.frag.end)
                curr_root.frag.start = 0
                curr_root.frag.end = len(curr_root.frag.graph.sent)
                #logger.writeln('str side is:' +  curr_root.frag.str_side())
                (hrg_rule, att_list_mapping) = self.extract_one_rule(curr_root, leafs, curr_ext_list, first) #This should return a ordered list of attachment nodes for each leaf
                if hrg_rule:
                    derivation_rules.append(hrg_rule)
                curr_root.frag.start = temp[0]
                curr_root.frag.end = temp[1]

            first = False
            if leafs is None:
                continue
            for child_node in leafs:
                if child_node.cut == 0: #This only appears when this child is the leaf
                    continue

                try:
                    assert child_node in att_list_mapping.keys(), 'There is no mapping from child node to its attachment list'
                except AssertionError:
                    logger.writeln('No mapping for sent %d' % self.sent_no)
                    #f.write('\n')
                    return None
                if child_node in att_list_mapping:
                    att_l = att_list_mapping[child_node]
                else:
                    att_l = []
                stack.append((child_node, curr_depth+4, att_l))
        return derivation_rules

    #This is to identify whether the root of the node has its concept in it
    #We have to verify if each external node has its concept in it
    def get_node_suffix(self, curr_node, ext_list):
        assert curr_node.cut == 1, 'The current node is not a cut node'
        amr_graph = curr_node.frag.graph
        edges = curr_node.frag.edges
        curr_root = curr_node.frag.root
        suffix = '1' if edges[amr_graph.nodes[curr_root].c_edge] == 1 else '0'
        for ext_id in ext_list:
            if ext_id == curr_root:
                continue
            suffix = '%s1' % suffix if edges[amr_graph.nodes[ext_id].c_edge] == 1 else '%s0' % suffix
        return suffix

    #This is to specify the detail label of each external node
    #Remain to be implemented
    def get_node_externals(self, curr_node, ext_list):
        amr_graph = curr_node.frag.graph
        edges = curr_node.frag.edges
        curr_root = curr_node.frag.root
        print curr_node.frag.ext_label

        suffix = curr_node.frag.ext_label[curr_root]
        for ext_id in ext_list:
            assert ext_id in curr_node.frag.ext_label, 'external index not found'
            suffix += '.%s' % curr_node.frag.ext_label[ext_id]
            #if ext_id == curr_root:
            #    continue
            #suffix = '%s1' % suffix if edges[amr_graph.nodes[ext_id].c_edge] == 1 else '%s0' % suffix
        return suffix

    def extract_one_rule(self, curr_node, children, ext_list, first_rule=False):
        assert curr_node.cut == 1, 'The current node is not a cut'

        new_rule = Rule()
        n_ext = len(ext_list)
        suffix = self.get_node_suffix(curr_node, ext_list)

        if curr_node.frag.root in curr_node.frag.ext_set or first_rule:
            new_rule.lhs = intern('[A%d-%s]' % (n_ext, suffix))
        else:
            new_rule.lhs = intern('[A%d-%s]' % (n_ext+1, suffix))

        nonterm_index = self.init_nonterm_map(children)

        #Initiate a mapping from external nodes to its index
        ext_mapping = self.init_new_ext_mapping(ext_list, curr_node.frag.root)

        att_list_mapping = {}
        self.initialize_new_graph_side(new_rule.e, curr_node, children, nonterm_index, ext_mapping, att_list_mapping)

        new_rule.f = self.get_string_side(curr_node, children, att_list_mapping)

        if curr_node.noprint and children is None:
            #print 'discarded rule: %s' % filter_vars(new_rule.dumped_format())
            return (None, att_list_mapping)

        if children:
            for child_node in children:
                if child_node.cut == 0 and child_node.noprint:
                    #print 'discarded rule: %s' % filter_vars(new_rule.dumped_format())
                    return (None, att_list_mapping)

        return (new_rule, att_list_mapping)

    #Derive all the composed SHRG rules in the current tree
    def dump_hrg_rules(self, f):
        root_node = self.hg.root
        assert len(root_node.frag.ext_set) == 0, 'The whole amr graph has external nodes'
        stack = [(root_node, 0, [])]
        result = ''
        while len(stack) > 0:
            curr_root, curr_depth, curr_ext_list = stack.pop()
            leafs = children(curr_root)
            (rule_str, att_list_mapping) = self.retrieve_one_hrg_rule(curr_root, leafs, curr_ext_list) #This should return a ordered list of attachment nodes for each leaf
            rule_str = ' ' * curr_depth + rule_str
            #logger.writeln(str(len(curr_ext)))
            #f.write('%s\n'% rule_str)
            result += '%s\n' % rule_str
            if leafs is None:
                continue
            for child_node in leafs:
                if child_node.cut == 0: #This only appears when this child is the leaf
                    continue

                try:
                    assert child_node in att_list_mapping.keys(), 'There is no mapping from child node to its attachment list'
                except AssertionError:
                    logger.writeln('No mapping for sent %d' % self.sent_no)
                    #f.write('\n')
                    return
                if child_node in att_list_mapping:
                    att_l = att_list_mapping[child_node]
                else:
                    att_l = []
                stack.append((child_node, curr_depth+4, att_l))
        f.write('%s\n' % result)
        #f.write('\n')

    def init_ext_mapping(self, ext_list):
        ext_mapping = {}
        for (i, ext_index) in enumerate(ext_list):
            ext_mapping[ext_index] = i
        return ext_mapping

    #Slight change, external nodes do not include root
    def init_new_ext_mapping(self, ext_list, curr_root):
        ext_mapping = {}
        index = 0
        for (i, ext_index) in enumerate(ext_list):
            if ext_index == curr_root:
                continue
            ext_mapping[ext_index] = index
            index += 1
        return ext_mapping

    #To verify if the children are ordered in their positions in the string
    def verify_order(self, children):
        start = -1
        end = -1
        for curr_node in children:
            curr_start = curr_node.frag.start
            if curr_start == -1: #Unaligned fragment
                continue
            curr_end = curr_node.frag.end
            assert curr_start > start, 'There are disordered pieces of fragments:\n(%d, %d)\n(%d, %d)' % (start, end, curr_start, curr_end)
            start = curr_start
            end = curr_end

    #init the nontermial mapping
    def init_nonterm_map(self, children):
        if children is None: #The current node is already the leaf
            return {}

        self.verify_order(children)
        nonterm_index = {}
        index = 0

        for child_node in children:
            if child_node.cut == 1: #Current node is a cut, so rewrite it with a nonterminal
                nonterm_index[child_node] = index
                index += 1
        return nonterm_index

    def get_string_side(self, curr_node, children, att_list_mapping):
        if children is None: #The current node is already the leaf
            return (curr_node.frag.str_list())

        sym_list = []
        #self.verify_order(children)
        end = -1
        graph = curr_node.frag.graph
        #nonterm_index = {}
        #index = 0

        end = curr_node.frag.start
        for child_node in children:
            curr_start = child_node.frag.start
            if curr_start != -1:
                if end != -1: #There are already some aligned fragments before
                    for unaligned_pos in xrange(end, curr_start):
                        try:
                            sym_list.append(graph.sent[unaligned_pos])
                            #s += graph.sent[unaligned_pos]
                            #s += ' '
                        except:
                            print graph.sent
                            print child_node.frag.start, child_node.frag.end, child_node.frag.graph_side({})
                            sys.exit(-1)

            if child_node.cut == 1: #Current node is a cut, so rewrite it with a nonterminal
                n_child_ext = len(child_node.frag.ext_set)
                suffix = self.get_node_suffix(child_node, att_list_mapping[child_node])
                if child_node.frag.root in child_node.frag.ext_set:
                    nonterm_sym = intern('[A%d-%s]' % (n_child_ext, suffix))
                else:
                    nonterm_sym = intern('[A%d-%s]' % (n_child_ext+1, suffix))
                sym_list.append(nonterm_sym)

                #nonterm_index[child_node] = index
                #index += 1
            else:
                #s += child_node.frag.str_side()
                #s += ' '
                sym_list.extend(child_node.frag.str_list())

            if curr_start != -1:
                #start = curr_start
                end = child_node.frag.end
        if curr_node.frag.end != end: #There are some unaligned tokens
            sym_list.extend(graph.sent[end:curr_node.frag.end])
        #assert curr_node.frag.end == end, 'The composition of fragments results in inconsitent positions'
        return sym_list

    def derive_string_side(self, curr_node, children):
        if children is None: #The current node is already the leaf
            return (curr_node.frag.str_side(), {})

        self.verify_order(children)
        s = ''
        #start = -1
        end = -1
        graph = curr_node.frag.graph
        nonterm_index = {}
        index = 0

        end = curr_node.frag.start
        for child_node in children:
            curr_start = child_node.frag.start
            if curr_start != -1:
                if end != -1: #There are already some aligned fragments before
                    for unaligned_pos in xrange(end, curr_start):
                        try:
                            s += graph.sent[unaligned_pos]
                            s += ' '
                        except:
                            print 'weird things happen on the string side'
                            print graph.sent
                            print child_node.frag.start, child_node.frag.end, child_node.frag.graph_side({})
                            sys.exit(-1)

            if child_node.cut == 1: #Current node is a cut, so rewrite it with a nonterminal
                s += '[A%d,%d] ' % (len(child_node.frag.ext_set), index)
                #print '%d %d %s' % (child_node.frag.start, child_node.frag.end, child_node.frag.str_side())
                nonterm_index[child_node] = index
                index += 1
            else:
                s += child_node.frag.str_side()
                s += ' '

            if curr_start != -1:
                #start = curr_start
                end = child_node.frag.end
        assert curr_node.frag.end == end, 'The composition of fragments results in inconsitent positions'
        return (s.strip(), nonterm_index)

    def get_graph_side(self, curr_node, children, ext_mapping, nonterm_index, att_list_mapping):
        if children is None:
            return curr_node.frag.graph_side(ext_mapping)

        #Now there are children of current node
        assert check_node_consist(curr_node, children), 'Unexpected parent seen with %s, from %d-%d' % (str(curr_node.frag), curr_node.frag.start, curr_node.frag.end)
        unvisited_nodes = set(children)
        var_mapping = {}
        rule_str = self.HRG_rule_under_node(curr_node.frag.root, ext_mapping, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping)
        try:
            assert len(unvisited_nodes) == 0, 'There are still unvisited nodes'
        except AssertionError:
            logger.writeln('unvisited nodes happen in sent %d' % self.sent_no)
            logger.writeln('parent')
            logger.writeln(curr_node.frag.graph_side({}))
            logger.writeln(str(curr_node.frag))
            logger.writeln('Parent cut is %d' % curr_node.cut)
            logger.writeln('All children')
            for node in children:
                logger.writeln(node.frag.graph_side({}))
                logger.writeln(str(node.frag))
                logger.writeln('Current child cut is %d' % node.cut)
            logger.writeln('unvisited children')
            for node in unvisited_nodes:
                logger.writeln(node.frag.graph_side({}))
                logger.writeln(str(node.frag))
                logger.writeln('Current cut %d' % node.cut)
            return ''
        return rule_str

    def derive_graph_side(self, curr_node, children, ext_mapping, nonterm_index, att_list_mapping):
        if children is None:
            return curr_node.frag.graph_side(ext_mapping)

        #Now there are children of current node
        assert check_node_consist(curr_node, children), 'Unexpected parent seen with %s, from %d-%d' % (str(curr_node.frag), curr_node.frag.start, curr_node.frag.end)
        unvisited_nodes = set(children)
        var_mapping = {}
        rule_str = self.HRG_rule_under_node(curr_node.frag.root, ext_mapping, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping)
        try:
            assert len(unvisited_nodes) == 0, 'There are still unvisited nodes'
        except AssertionError:
            logger.writeln('unvisited nodes happen in sent %d' % self.sent_no)
            logger.writeln('parent')
            logger.writeln(curr_node.frag.graph_side({}))
            logger.writeln(str(curr_node.frag))
            logger.writeln('Parent cut is %d' % curr_node.cut)
            logger.writeln('All children')
            for node in children:
                logger.writeln(node.frag.graph_side({}))
                logger.writeln(str(node.frag))
                logger.writeln('Current child cut is %d' % node.cut)
            logger.writeln('unvisited children')
            for node in unvisited_nodes:
                logger.writeln(node.frag.graph_side({}))
                logger.writeln(str(node.frag))
                logger.writeln('Current cut %d' % node.cut)
            #sys.exit(-1)
        return rule_str


    def HRG_rule_under_node(self, curr_root, ext_mapping, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping, include_root=True):
        curr_node_str = ''
        if include_root:
            #Initiate the node variable notation
            if curr_root in var_mapping.keys():
                curr_node_str = 'n%d.' % var_mapping[curr_root]
            else:
                curr_node_str = 'n%d.' % var_mapping.setdefault(curr_root, len(var_mapping))

            if curr_root in ext_mapping.keys():
                curr_node_str += '*%d' % ext_mapping[curr_root]

        #Find a set of candidate nodes having the current index as their roots
        candidates = set()
        for node in unvisited_nodes.copy():
            if node.frag.root == curr_root:
                candidates.add(node)
                unvisited_nodes.remove(node)

        if len(candidates) == 0:
            return curr_node_str

        for cand_node in candidates:
            #We only want to extract the attachment node list for each nonterminal edges
            if cand_node.cut == 1: #means the current node is a nonterminal edge
                assert cand_node in nonterm_index.keys(), 'There are unappeared nonterminal on the HRG side'
                edge_label = '[A%d,%d]' % (len(cand_node.frag.ext_set), nonterm_index[cand_node])
                curr_node_str += ' :%s' % edge_label
                curr_att_list = []
                if curr_root in cand_node.frag.ext_set:
                    curr_att_list.append(curr_root)
                for curr_ext_index in cand_node.frag.ext_set:
                    if curr_ext_index == curr_root:
                        continue
                    curr_att_list.append(curr_ext_index)
                    ext_node_str = self.HRG_rule_under_node(curr_ext_index, ext_mapping, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping)
                    if ':' in ext_node_str:
                        curr_node_str += ' (%s)' % ext_node_str
                    else:
                        curr_node_str += ' %s' % ext_node_str
                assert cand_node not in att_list_mapping, 'Repeat key found'
                att_list_mapping[cand_node] = curr_att_list
            else:  #which means the current node should be rewritten as terminal fragments having the current node as root
                try:
                    assert cand_node not in nonterm_index.keys(), 'A terminal node found on the string side'
                except AssertionError:
                    logger.writeln('Nonterm invalid mapping happen in sent %d' % self.sent_no)

                cand_node_str = cand_node.frag.graph_side(ext_mapping, False, self, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping) #This representation does not have the root in it
                #assert len(cand_node_str) > 2, 'The fragment is too short: %s' % cand_node_str
                curr_node_str += ' %s' % cand_node_str

        return curr_node_str

    #This is to guarantee the order really corresponds to the order of attachment nodes
    #The external nodes of the composed fragment is explicit
    #The external nodes of smaller components are extracted at the rule extraction phase
    def retrieve_one_hrg_rule(self, curr_node, children, ext_list):
        assert curr_node.cut == 1, 'The current node is not a cut'
        #ext_set = curr_node.frag.ext_set
        n_ext = len(ext_list)

        lhs = '[A%d]' % n_ext
        #Retrieve the string side of the rule
        #Also a mapping from each nonterminal to it's index
        (rhs1, nonterm_index) = self.derive_string_side(curr_node, children)

        #Initiate a mapping from external nodes to its index
        ext_mapping = self.init_ext_mapping(ext_list)

        att_list_mapping = {}
        rhs2 = self.derive_graph_side(curr_node, children, ext_mapping, nonterm_index, att_list_mapping)

        return ('%s ||| %s ||| %s' % (lhs, rhs1, rhs2), att_list_mapping)

    '''
    xpeng: calculate the density factor for each incoming edge(each denoting a different sub-tree, thus density factor
    then choose an incoming edge according to density_factor(deg(v)) * posterior prob of each rule list(Pt(t))
    discount the choices that are dependent before and count the choices after, set the parent info after changing the edge
    if the sub-tree structure has changed, need to update the CUT-INDEX info
    '''
    def sample_edge(self, node, parent):
        if parent is None:  # root node
            parent = node

        if len(node.incoming) <= 1:  # trivial case
            return
        rule_lists = []
        old = node.edge

        density_factors = []
        for i in range(len(node.incoming)):
            node.edge = i
            rule_lists.append([r for n, r in self.composed_rules_under(parent)])
            density_factors.append(density_factor(node))

        i = choice(rule_lists, old, self.SAMPLER, density_factors)

        node.edge = i

        # update parent pointers
        set_parents_under(node)

    '''
    xpeng: the cut index is calculated using the rule lists prob by setting the cut index to 1 or 0, and compare the posterior
    '''
    def sample_cut(self, node, parent):
        if parent is None:  # root node
            return
        #if node.fj - node.fi > FLAGS.maxcut or node.ej - node.ei > FLAGS.maxcut:
        #    return
        rule_lists = []
        old = node.cut

        node.cut = 0
        rule_lists.append([self.make_composed_rule(parent)])
        node.cut = 1
        rule_lists.append([self.make_composed_rule(parent), self.make_composed_rule(node)])

        i = choice(rule_lists, old, self.SAMPLER)
        node.cut = i

    '''
    xpeng: smaller grained sites, in contrast to (Sample, Node) sites
    Each site is a node and its ancestral cut node
    '''
    def sites(self):
        queue = [(self.hg.root, None)]
        while len(queue) > 0:
            node, parent = queue.pop(0)
            yield node, parent
            if node.cut:
                parent = node
            if FLAGS.sample_cut_only:
                for child in children(node):
                    queue.append((child, parent))
            else:
                for child in node.incoming[node.edge].tail:
                    queue.append((child, parent))

    def sample(self):
        queue = [(self.hg.root, None)]
        while len(queue) > 0:
            node, parent = queue.pop(0)
            if FLAGS.sample_level is None or node.level <= FLAGS.sample_level:
                if node.nosample: #newly changed
                    node.cut = 0 #Never allow nonterminals greater than a threshold
                else:
                    if FLAGS.sample_edge:
                        self.sample_edge(node, parent)
                    if FLAGS.sample_cut:
                        self.sample_cut(node, parent)

            if node.cut:
                parent = node
            if FLAGS.sample_cut_only:
                mychildren = children(node)
                if mychildren is not None:
                    for child in children(node):
                        queue.append((child, parent))
            else:
                if len(node.incoming) > 0:
                    for child in node.incoming[node.edge].tail:
                        if child.nosample:
                            child.cut = 0
                            continue
                        queue.append((child, parent))

    def __str__(self):
        return self.str_helper_expand(self.hg.root)

    def str_helper(self, node, indent=0):
        result = ''
        rule = self.make_composed_rule(node)
        result += ' '*indent + str(rule) + ' ' +  str(node) + '\n'
        for child in children(node):
            result += self.str_helper(child, indent + 4)
        return result

    def str_helper_expand(self, node, indent=0):
        result = ''
        rule = self.make_one_level_rule(node)
        result += ' '*indent + str(rule) + ' ' +  str(node) + ' ' + ('cut: %s' % node.cut) + '\n'
        for child in node.incoming[node.edge].tail:
            result += self.str_helper_expand(child, indent + 4)
        return result

'''
xpeng: set the parent attribute for each node under the sub-tree rooted at the current node
the current tree, not the forest
'''
def set_parents_under(node):
    if node.edge < 0:
        return
    for child in node.incoming[node.edge].tail:
        child.parent = node
        set_parents_under(child)

def children(node):
    if len(node.incoming) == 0:
        return None
    edge = node.incoming[node.edge]
    result = []
    for n in edge.tail:
        if n.cut or (len(n.incoming) == 0):
            result.append(n)
        else:
            result.extend(children(n))
    return result

def density_factor(node):
    "How many choices are there under this node?"
    selected_nodes = nodes_turned_on_under(node)
    #for node in selected_nodes:
    #    print(node, len(node.incoming))
    result = 1
    for n in selected_nodes:
        result *= len(n.incoming)
    assert result > 0, 'Meeting some 0 length of incoming nodes'
    result *= 2**len(selected_nodes)
    return result

def cut_nodes_under(node):
    "pre-order, self not included"
    for child in children(node):
        yield child
        for c in cut_nodes_under(child):
            yield c

def nodes_turned_on_under(node):
    result = []
    queue = [node]
    while len(queue) > 0:
        curr = queue.pop(0)
        if len(curr.incoming) > 0:
            result.append(curr)
            for child in curr.incoming[curr.edge].tail:
                queue.append(child)
    return result

def nodes_in_fragment_under(node):
    "return all nodes that are in the fragment marked by cut points, including self, pre-order"
    yield node
    if not node.cut:
        if len(node.incoming) > 0:
            for n in node.incoming[node.edge].tail:
                for n1 in nodes_in_fragment_under(n):
                    yield n1

def check_node_consist(parent, children):
    try:
        nodes = children[0].frag.nodes | children[1].frag.nodes
        edges = children[0].frag.edges | children[1].frag.edges
    except:
        print 'incomplete parent and children'
        print str(parent.frag)
        for child in children:
            print str(node.frag)
        assert False, 'Here comes the error'
    for i in xrange(2, len(children)):
        nodes |= children[i].frag.nodes
        edges |= children[i].frag.edges
    if parent.frag.nodes != nodes or parent.frag.edges != edges:
        return False
    return True
