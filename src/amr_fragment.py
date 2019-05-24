#!/usr/bin/env python2.7
import sys
import bitarray
import copy
from bitarray import bitarray
from HRGSample import Sample
#Each fragment is simply two vectors of bits, for edges and boundary nodes
class AMRFragment(object):
    def __init__(self, n_edges, n_nodes, graph):
        self.edges = bitarray(n_edges)
        if self.edges.count() != 0:
            self.edges ^= self.edges
        assert self.edges.count() == 0, 'initialization nonzero'
        self.nodes = bitarray(n_nodes)
        if self.nodes.count() != 0:
            self.nodes ^= self.nodes
        assert self.nodes.count() == 0, 'initialization nonzero'
        self.graph = graph
        self.root = -1
        self.roots = set() #This is used to allow multiple root structure
        self.ext_set = set()
        self.ext_list = []
        self.ext_mapping = {}
        self.start = -1
        self.end = -1
        self.unaligned = []
        self.ext_label = {} #This attribute is used for the refinement of the external nodes


    def edge_count(self):
        return self.edges.count()

    def all_edges(self):
        return [i for i in xrange(len(self.edges)) if self.edges[i] == 1]

    def set_span(self, start, end):
        self.start = start
        self.end = end

    def set_root(self, node_num):
        self.set_node(node_num)
        self.root = node_num

    def str_side(self):
        if self.start == -1:
            return ''
        return ' '.join(self.graph.sent[self.start:self.end])

    def str_list(self):
        if self.start == -1:
            return []
        return self.graph.sent[self.start:self.end]

    def init_hgraph(self, hgraph, node_index, var_mapping, visited_index, sample=None, nonterm_index=None, unvisited_nodes=None):
        assert node_index not in visited_index, 'Initiate another visit for a visited node index'
        graph = self.graph
        curr_node = graph.nodes[node_index]
        curr_ident = None
        if node_index not in var_mapping: #else it's already visited in some nonterminal
            curr_ident = 'N%d' % var_mapping.setdefault(node_index, len(var_mapping))
            root_label = '*'
            ignoreme = hgraph[curr_ident] #build an ident for current node

        curr_ident = 'N%d' % var_mapping[node_index]
        assert curr_ident in hgraph, 'Weird, some node is not verified'
        if len(visited_index) == 0:
            hgraph.roots.append(curr_ident)
        visited_index.add(node_index)

        const_edge_index = curr_node.c_edge
        assert curr_ident is not None, 'have not specified the current node identifier'

        if self.edges[const_edge_index] == 1: #The current node has some terminal notation attached to it
            hgraph.num_edges += 1
            curr_edge_id = curr_node.node_label() #Identifier
            if curr_node.is_var_node():
                curr_edge_id = graph.dict[curr_edge_id] #Label
            child = tuple() #Initialize a empty tuple for a leaf edge
            hgraph._add_triple(curr_ident, curr_edge_id, child)

        if curr_node.is_leaf():
            return

        for edge_index in curr_node.v_edges:
            curr_edge = graph.edges[edge_index]
            if self.edges[edge_index] == 1:
                hgraph.num_edges += 1
                curr_edge_id = curr_edge.label
                tail_node_index = curr_edge.tail
                tail_node_ident = None
                if tail_node_index not in var_mapping:
                    tail_node_ident = 'N%d' % var_mapping.setdefault(tail_node_index, len(var_mapping))
                tail_node_ident = 'N%d' % var_mapping[tail_node_index]
                ignoreme = hgraph[tail_node_ident]
                hgraph._add_triple(curr_ident, curr_edge_id, (tail_node_ident,))
                if tail_node_index not in visited_index:
                    self.init_hgraph(hgraph, tail_node_index, var_mapping, visited_index, sample, nonterm_index, unvisited_nodes) #Further search for lower subgraph

                if (tail_node_index in self.ext_set) and sample is not None:
                    sample.graph_under_node(tail_node_index, hgraph, var_mapping, nonterm_index, unvisited_nodes)

    def init_new_hgraph(self, hgraph, node_index, var_mapping, visited_index, sample=None, nonterm_index=None, unvisited_nodes=None, ext_mapping=None, att_list_mapping=None, is_root=False):
        assert node_index not in visited_index, 'Initiate another visit for a visited node index'
        graph = self.graph
        curr_node = graph.nodes[node_index]
        curr_ident = None
        if node_index not in var_mapping: #else it's already visited in some nonterminal
            curr_ident = '_%d' % var_mapping.setdefault(node_index, len(var_mapping))
            #root_label = '*'
            ignoreme = hgraph[curr_ident] #build an ident for current node
            if node_index in ext_mapping:
                ext_id = ext_mapping[node_index]
                if curr_ident in hgraph.external_nodes and hgraph.external_nodes[curr_ident] != ext_id:
                    assert False, 'external nodes id does not match'
                hgraph.external_nodes[curr_ident] = ext_id
                hgraph.rev_external_nodes[ext_id] = curr_ident

        curr_ident = '_%d' % var_mapping[node_index]
        assert curr_ident in hgraph, 'Weird, some node is not verified'
        if len(visited_index) == 0 and len(hgraph.roots) == 0:
            hgraph.roots.append(curr_ident)
        visited_index.add(node_index)

        const_edge_index = curr_node.c_edge
        assert curr_ident is not None, 'have not specified the current node identifier'

        if self.edges[const_edge_index] == 1: #The current node has some terminal notation attached to it
            hgraph.num_edges += 1
            #curr_edge_id = curr_node.node_label() #Identifier
            #if curr_node.is_var_node():
            #    curr_edge_id = graph.dict[curr_edge_id] #Label
            curr_edge_id = str(curr_node)
            if '/' in curr_edge_id:
                if curr_edge_id.split('/')[0].strip() in '0123456789':
                    curr_edge_id = curr_edge_id.replace('/', '@@@@') #Here is a special trick, treat each / symbol in const as @@@@
            child = tuple() #Initialize a empty tuple for a leaf edge
            hgraph._add_triple(curr_ident, curr_edge_id, child)

        if curr_node.is_leaf():
            return

        for edge_index in curr_node.v_edges:
            curr_edge = graph.edges[edge_index]
            if self.edges[edge_index] == 1:
                hgraph.num_edges += 1
                curr_edge_id = curr_edge.label
                tail_node_index = curr_edge.tail
                tail_node_ident = None
                if tail_node_index not in var_mapping:
                    tail_node_ident = '_%d' % var_mapping.setdefault(tail_node_index, len(var_mapping))
                tail_node_ident = '_%d' % var_mapping[tail_node_index]
                ignoreme = hgraph[tail_node_ident]
                if tail_node_index in ext_mapping:
                    ext_id = ext_mapping[tail_node_index]
                    if tail_node_ident in hgraph.external_nodes and hgraph.external_nodes[tail_node_ident] != ext_id:
                        assert False, 'external nodes id does not match'
                    hgraph.external_nodes[tail_node_ident] = ext_id
                    hgraph.rev_external_nodes[ext_id] = tail_node_ident
                hgraph._add_triple(curr_ident, curr_edge_id, (tail_node_ident,))
                if tail_node_index not in visited_index:
                    self.init_new_hgraph(hgraph, tail_node_index, var_mapping, visited_index, sample, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping) #Further search for lower subgraph

                if (tail_node_index in self.ext_set) and sample is not None:
                    sample.new_graph_under_node(tail_node_index, hgraph, var_mapping, nonterm_index, unvisited_nodes, ext_mapping, att_list_mapping)

    def graph_side(self, ext_mapping, include_root=True, sample=None, var_mapping=None, nonterm_index=None, unvisited_nodes=None, att_list_mapping=None):
        visited_index = set()
        return self.sub_graph_under(self.root, ext_mapping, include_root, sample, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping, visited_index)

    def sub_graph_under(self, node_index, ext_mapping, include_root=True, sample=None, var_mapping=None, nonterm_index=None, unvisited_nodes=None, att_list_mapping=None, visited_index=None):
        result = ''
        if include_root:
            result = '.'
            if node_index in ext_mapping.keys():
                result += '*%d' % ext_mapping[node_index]

        visited_index.add(node_index)

        graph = self.graph
        curr_node = graph.nodes[node_index]
        const_edge_index = curr_node.c_edge

        if self.edges[const_edge_index] == 1:
            result = '%s :%s' % (result, str(curr_node))

        if curr_node.is_leaf():
            return result.strip()

        for edge_index in curr_node.v_edges:
            curr_edge = graph.edges[edge_index]
            if self.edges[edge_index] == 1:
                #if True:
                if curr_edge.tail in visited_index:
                    sub_result = str(self.graph.edges[self.graph.nodes[curr_edge.tail].c_edge])
                else:
                    sub_result = self.sub_graph_under(curr_edge.tail, ext_mapping, True, sample, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping, visited_index)
                    if ':' in sub_result:
                        sub_result = '(%s)' % sub_result
                result += ' :%s %s' % (str(curr_edge), sub_result)
                if (curr_edge.tail not in visited_index) and (curr_edge.tail in self.ext_set) and sample is not None:
                    ext_node_str = sample.HRG_rule_under_node(curr_edge.tail, ext_mapping, var_mapping, nonterm_index, unvisited_nodes, att_list_mapping, False)
                    if ext_node_str != '':
                        result += ext_node_str

        return result.strip()

    def add_ext_node(self, node_num):
        self.ext_set.add(node_num)

    def set_edges(self, edges):
        self.edges = edges

    def set_nodes(self, nodes):
        self.nodes = nodes

    def set_edge(self, edge_num):
        self.edges[edge_num] = 1

    def set_node(self, node_num):
        self.nodes[node_num] = 1

    def set_ext_set(self, ext_set):
        self.ext_set = ext_set

    def node_list(self):
        return [i for (i, existed) in enumerate(self.nodes) if existed]

    def edge_list(self):
        return [i for (i, existed) in enumerate(self.edges) if existed]

    def ext_nodes_str(self):
        s = ''
        for node_index in self.ext_set:
            s += str(self.graph.nodes[node_index])
            s += ' '
        return s.strip()

    #def create_ext_mapping(self):
    #    #num = 0
    #    for node_index in self.ext_set:
    #        self.ext_mapping[node_index] = '*%d' % num
    #        num += 1

    @staticmethod
    def initialize_from_alignment(nodes, edges, graph=None):
        frag = AMRFragment(len(edges), len(nodes), graph)
        frag.edges = edges
        frag.nodes = nodes
        return frag

    def __eq__(self, other):
        return ((self.edges == other.edges) and (self.nodes == other.nodes))

    def __hash__(self):
        s = ''
        if self.nodes != None:
            s += str(self.nodes)
        s += '-'
        if self.edges != None:
            s += str(self.edges)
        return hash(s)

    #def retrieve_sub_graph(self):
    #    assert self.root != -1, 'The root node is not specified'
    #    visited_index = set()
    #    return self.sub_graph_node(self.root, visited_index)

    def is_ext(self, node_num):
        curr_node = self.graph.nodes[node_num]
        for edge_index in curr_node.edge_set():
            if self.edges[edge_index] == 0:
                return True
        return False

    def frag_repr(self, node_index, visited_index):
        assert node_index not in visited_index
        visited_index.add(node_index)

        result = '.'
        node = self.graph.nodes[node_index]
        const_edge_index = node.c_edge

        if self.edges[const_edge_index] == 1:
            result = '%s :%s' % (result, str(node))

        incoming_edges = []
        for edge_index in node.v_edges:
            if self.edges[edge_index] == 0:
                continue
            incoming_edges.append(edge_index)

        if len(incoming_edges) > 1:
            incoming_edges = sorted(incoming_edges, key=lambda index: str(self.graph.edges[index]))

        for edge_index in incoming_edges:
            curr_edge = self.graph.edges[edge_index]
            if curr_edge.tail in visited_index:
                tail_node = self.graph.nodes[curr_edge.tail]
                if self.edges[tail_node.c_edge] == 1:
                    result = '%s :%s (%s)' % (result, str(curr_edge), str(self.graph.edges[self.graph.nodes[curr_edge.tail].c_edge]))
                else:
                    result = '%s :%s .' % (result, str(curr_edge))
            else:
                sub_frag_repr = self.frag_repr(curr_edge.tail, visited_index)
                if ':' in sub_frag_repr:
                    sub_frag_repr = '(%s)' % sub_frag_repr
                result = '%s :%s %s' % (result, str(curr_edge), sub_frag_repr)
        return result

    def build_ext_set(self):
        self.ext_set = set(self.ext_list)

    #Perform dfs in a sorted way
    def build_ext_list(self):

        self.ext_list = []
        assert self.root != -1, 'Root node not designated'
        visited_index = set()

        stack = [self.root]

        while stack:
            curr_node_index = stack.pop()
            assert self.nodes[curr_node_index] == 1
            if curr_node_index in visited_index: #Newly added, there might be reentrancy to same node between two nodes
                continue

            visited_index.add(curr_node_index)

            if self.is_ext(curr_node_index):
                self.ext_list.append(curr_node_index)

            curr_node = self.graph.nodes[curr_node_index]

            incoming_edges = []
            for edge_index in curr_node.v_edges:
                if self.edges[edge_index] == 0:
                    continue

                curr_edge = self.graph.edges[edge_index]
                if curr_edge.tail in visited_index: #The tail node is already visited
                    continue

                incoming_edges.append((str(curr_edge), edge_index))

            if len(incoming_edges) > 1:
                incoming_edges = sorted(incoming_edges, key=lambda elem: elem[0])
            for (edge_label, edge_index) in reversed(incoming_edges):
                curr_edge = self.graph.edges[edge_index]
                tail_index = curr_edge.tail
                stack.append(tail_index)

        for (i, ext_index) in enumerate(self.ext_list):
            self.ext_mapping[ext_index] = i

    #Use with caution, only for smallest grained fragments
    def single_exp(self):
        stack = [self.root]
        amr_graph = self.graph
        curr_list = []
        while stack:
            curr_index = stack.pop()
            curr_node = amr_graph.nodes[curr_index]
            curr_list.append(curr_node.node_str())
            if curr_node.node_str() == 'date-entity':
                return 'date-entity'
            for edge_index in curr_node.v_edges:
                if self.edges[edge_index] == 1:
                    curr_edge = amr_graph.edges[edge_index]
                    edge_str = str(curr_edge)
                    if edge_str[:2] == 'op':
                        continue
                    stack.append(curr_edge.tail)

        return '+'.join(curr_list)

    def __str__(self):
        visited_index = set()
        return self.frag_repr(self.root, visited_index)

    def missing_edges(self):
        s = ""
        for i in xrange(len(self.edges)):
            if self.edges[i] == 0:
                s += self.graph.edges[i].label
                s += '  '
        return s

    def var_from_graph(amr_graph, binary_reps):
        bits = binary_rep.split('.')
        assert bits[0] == '0', 'All binary representation should start from root 0'

def check_adjacent(f1, f2):
    result = f1.nodes & f2.nodes
    return result.count() != 0

def connect_adjacent(frags, logger):
    visited = set()
    new_frag_list = []
    updated = True
    for frag in frags:
        if frag in visited:
            continue
        visited.add(frag)
        curr_result = frag
        updated = False
        for cand_frag in frags:
            if cand_frag in visited:
                continue
            if check_disjoint(curr_result, cand_frag) and check_adjacent(curr_result, cand_frag):

                new_result = combine_fragments(curr_result, cand_frag)
                if new_result:
                    updated = True
                    visited.add(cand_frag)
                    visited.add(curr_result)
                    curr_result = new_result
                else:
                    logger.writeln("Before:")
                    logger.writeln(str(cand_frag))
                    logger.writeln(str(curr_result))
                    logger.writeln("After:")
                    logger.writeln(str(new_result))

        new_frag_list.append(curr_result)
    return new_frag_list

#If two fragments are disjoint, then they do not share any common edge
def check_disjoint(f1, f2):
    result = f1.edges & f2.edges
    return result.count() == 0

#Root operation is a bold guess: that the root of combination must be a root in one of the child fragment
def combine_fragments(f1, f2, refine=False):
    f1_rooted = (f2.root in f1.ext_set)
    f2_rooted = (f1.root in f2.ext_set)

    assert not refine
    #Currently we don't allow combination of subgraph having two roots
    if not (f1_rooted or f2_rooted):
        return None

    n_nodes = len(f1.nodes)
    n_edges = len(f1.edges)
    amr_graph = f1.graph
    new_frag = AMRFragment(n_edges, n_nodes, amr_graph)

    if f1.root != f2.root and f1_rooted and f2_rooted: #There is a cycle
        f1_rootnode = amr_graph.nodes[f1.root]
        f2_rootnode = amr_graph.nodes[f2.root]

        new_frag.root = f2.root if len(f2_rootnode.p_edges) > len(f1_rootnode.p_edges) else f1.root
    elif f1_rooted:
        new_frag.root = f1.root
    else:
        new_frag.root = f2.root
    nodes = f1.nodes | f2.nodes
    edges = f1.edges | f2.edges

    new_frag.set_edges(edges)
    new_frag.set_nodes(nodes)

    new_frag.build_ext_list()
    new_frag.build_ext_set()

    #Setting the word span of the new fragment
    if f1.start == -1:
        new_frag.set_span(f2.start, f2.end)
    elif f2.start == -1:
        new_frag.set_span(f1.start, f1.end)
    else:
        assert f1.start < f2.start, 'Should align according to order'
        if f1.start < f2.start:
            assert f1.end <= f2.start, 'overlapping fragments'
            new_frag.set_span(f1.start, f2.end)
        else:
            new_frag.set_span(f2.start, f1.end)
    return new_frag

def find_unaligned_edge(curr_index, another_index, amr_graph, edge_alignment):
    curr_node = amr_graph.nodes[curr_index]

    n_nodes = len(amr_graph.nodes)
    n_edges = len(amr_graph.edges)
    unaligned_frag = AMRFragment(n_edges, n_nodes, amr_graph)

    for edge_index in curr_node.p_edges:
        if edge_alignment[edge_index] == 1:
            continue
        curr_pedge = amr_graph.edges[edge_index]
        p_index = curr_pedge.head

        if p_index == another_index:
            unaligned_frag.set_root(p_index)

            unaligned_frag.set_node(p_index)
            unaligned_frag.set_node(curr_index)

            unaligned_frag.set_edge(edge_index)

            unaligned_frag.build_ext_list()
            unaligned_frag.build_ext_set()
            return unaligned_frag

    return None

def find_unaligned_path(curr_index, frag, edge_alignment, refine=False):
    amr_graph = frag.graph
    curr_node = amr_graph.nodes[curr_index]
    path = []

    n_nodes = len(amr_graph.nodes)
    n_edges = len(amr_graph.edges)
    unaligned_frag = AMRFragment(n_edges, n_nodes, amr_graph)

    for edge_index in curr_node.p_edges:
        if edge_alignment[edge_index] == 1:
            continue
        curr_pedge = amr_graph.edges[edge_index]
        p_index = curr_pedge.head

        #If two fragments are connected through one relation
        if p_index in frag.ext_set:
            ext_set = set()
            unaligned_frag.set_root(p_index)

            unaligned_frag.set_node(p_index)
            unaligned_frag.set_node(curr_index)

            unaligned_frag.set_edge(edge_index)

            unaligned_frag.build_ext_list()
            unaligned_frag.build_ext_set()
            if refine:
                init_ext_frag(unaligned_frag)
            return unaligned_frag

    return None

def find_common_roots(root1, root2, amr_graph, edge_alignment):

    n_nodes = len(amr_graph.nodes)
    n_edges = len(amr_graph.edges)
    unaligned_frag = AMRFragment(n_edges, n_nodes, amr_graph)

    #From the parent of root1, find a path to root2
    curr_node = amr_graph.nodes[root1]
    for edge_index in curr_node.p_edges:
        if edge_alignment[edge_index] == 1:
            continue
        curr_pedge = amr_graph.edges[edge_index]
        p_index = curr_pedge.head
        p_node = amr_graph.nodes[p_index]

        if edge_alignment[p_node.c_edge] == 1:
            return None

        for sib_edge_index in p_node.v_edges:
            if sib_edge_index == edge_index or edge_alignment[sib_edge_index] == 1:
                continue
            curr_sedge = amr_graph.edges[sib_edge_index]
            if curr_sedge.tail == root2:
                unaligned_frag.set_root(p_index)

                unaligned_frag.set_node(p_index)
                unaligned_frag.set_node(root1)
                unaligned_frag.set_node(root2)

                unaligned_frag.set_edge(p_node.c_edge)
                unaligned_frag.set_edge(edge_index)
                unaligned_frag.set_edge(sib_edge_index)

                unaligned_frag.build_ext_list()
                unaligned_frag.build_ext_set()
                return unaligned_frag
    return None

#When dealing with cycles and reentrancies, there
#might be some edges that are not aligned
def connect_all_internal_edges(f, edge_alignment, refine=False):
    amr_graph = f.graph
    n_nodes = len(amr_graph.nodes)
    n_edges = len(amr_graph.edges)

    unalign_frags = []
    for first_index in f.ext_set:
        for second_index in f.ext_set:
            if first_index == second_index:
                continue
            first_node = amr_graph.nodes[first_index]
            for edge_index in first_node.v_edges:
                curr_edge = amr_graph.edges[edge_index]
                if (curr_edge.tail == second_index) and edge_alignment[edge_index] == 0 and f.edges[edge_index] == 0:
                    unaligned_frag = AMRFragment(n_edges, n_nodes, amr_graph)
                    unaligned_frag.set_root(first_index)
                    unaligned_frag.set_node(first_index)
                    unaligned_frag.set_node(second_index)
                    unaligned_frag.set_edge(edge_index)
                    unaligned_frag.build_ext_list()
                    unaligned_frag.build_ext_set()

                    if refine:
                        init_ext_frag(unaligned_frag, False, False)

                    unalign_frags.append(unaligned_frag)

    return unalign_frags

#Find the combination of two fragments which are not necessarily connected
#There should be a unaligned path from the root of one fragment to a boundary node of another fragment
def general_combine_fragments(f1, f2, edge_alignment, refine=False):
    new_frag = combine_fragments(f1, f2, refine)

    #Cycle detected
    if new_frag:
        inter_frags = connect_all_internal_edges(new_frag, edge_alignment, refine)
        for frag in inter_frags:
            new_frag = combine_fragments(new_frag, frag, refine)
        return (new_frag, inter_frags)

    f1_root_path = find_unaligned_path(f2.root, f1, edge_alignment, refine)
    if f1_root_path:
        above_frag = combine_fragments(f1, f1_root_path, refine)
        new_frag = combine_fragments(above_frag, f2, refine)
        assert new_frag.root == f1.root
        inter_frags = connect_all_internal_edges(new_frag, edge_alignment, refine)
        for frag in inter_frags:
            new_frag = combine_fragments(new_frag, frag, refine)
        return (new_frag, [f1_root_path] + inter_frags)

    f2_root_path = find_unaligned_path(f1.root, f2, edge_alignment, refine)
    if f2_root_path:
        above_frag = combine_fragments(f2, f2_root_path, refine)
        new_frag = combine_fragments(f1, above_frag, refine)
        assert new_frag.root == f2.root
        inter_frags = connect_all_internal_edges(new_frag, edge_alignment, refine)
        for frag in inter_frags:
            new_frag = combine_fragments(new_frag, frag, refine)
        return (new_frag, [f2_root_path] + inter_frags)

    #common_root_path = find_common_roots(f1.root, f2.root, f1.graph, edge_alignment)
    #if common_root_path:
    #    first_frag = combine_fragments(common_root_path, f1)
    #    new_frag = combine_fragments(first_frag, f2)
    #    inter_frags = connect_all_internal_edges(new_frag, edge_alignment)
    #    for frag in inter_frags:
    #        new_frag = combine_fragments(new_frag, frag)
    #    return (new_frag, [common_root_path] + inter_frags)
    return (None, None)

def check_consist(parent, children):
    nodes = children[0].frag.nodes | children[1].frag.nodes
    edges = children[0].frag.edges | children[1].frag.edges
    for i in xrange(2, len(children)):
        nodes |= children[i].frag.nodes
        edges |= children[i].frag.edges
    if parent.frag.nodes != nodes or parent.frag.edges != edges:
        return False
    return True

#Initialize the external mapping for the initialize fragments
def init_ext_frag(frag, is_pred=False, is_op=False):

    #First initialize the root
    amr_graph = frag.graph
    root_node = amr_graph.nodes[frag.root]
    if frag.edges[root_node.c_edge] == 0:
        frag.ext_label[frag.root] = '.' #If a fragment does not have a root label, the root is identified as dot

    else:
        root_label = str(root_node)
        #if 'ARG' in str(frag) and ('of' not in str(frag)) and '-' in root_label: #Indicate we are dealing with a predicate structure
        if is_pred:
            frag.ext_label[frag.root] = 'PRED'
        elif is_op:
            frag.ext_label[frag.root] = root_label if '/' not in root_label else root_label.split('/')[1]
        else:
            frag.ext_label[frag.root] = 'ENT'

    for ext_index in frag.ext_set: #I assume only could be nodes without concept
        if ext_index == frag.root:
            continue

        curr_node = amr_graph.nodes[ext_index]
        assert frag.edges[curr_node.c_edge] == 0, 'weird node with concept found'

        #assert len(amr_graph.nodes[ext_index].p_edges) == 1, 'not only 1 parent found'
        for p_edge_index in curr_node.p_edges:
            if frag.edges[p_edge_index] == 1:
                curr_edge_label = str(amr_graph.edges[p_edge_index])
                frag.ext_label[ext_index] = curr_edge_label


