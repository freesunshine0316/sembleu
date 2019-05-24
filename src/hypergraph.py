#!/usr/bin/env python3
import os
import sys

import gflags
FLAGS = gflags.FLAGS

from common import INF
from logprob import logsum, logprod, LOGZERO
import logger

#modified to the settings of AMR
gflags.DEFINE_boolean(
    'minimize_path_cost',
    False,
    'This fix the bug that path with cost closest to 0 is considered best.')

def escape_quote(s):
    return s.replace('"', r'\"')

class Node(object):
    def __init__(self):
        self.incoming = []
        self.nout = 0  # number of outgoing edges
        self.hg = None  # hypergraph this node belongs to
        self.best_paths_list = None

    def add_incoming(self, edge):
        self.incoming.append(edge)
        edge.head = self

    def id_str(self):
        """The id of a node is assigned after topological sort in reversed
        topological order. (Root has id 0.) Use python object id if this node
        is not assigned a id"""
        if hasattr(self, 'id'):
            return str(self.id)
        else:
            return 'obj%s' % id(self)

    def dot_label(self, detailed = True):
        """Returns label used in dot representation."""
        if detailed:
            return '%s: %s' % (self.id_str(), escape_quote(str(self)))
        else:
            return '%s' % self.id_str()

    def dot(self, color='', detailed=True):
        """dot language representation of this node and its incoming edges"""
        result = 'n%s [label="%s" style="filled" color="%s"];\n' % \
                (self.id_str(),
                 self.dot_label(detailed=detailed),
                 color)
        # write hyperedges
        for i, edge in enumerate(self.incoming):
            edgename = 'e%s_%s' % (self.id_str(), i)
            # graph node for hyperedge
            result += '%s [shape="point"]\n' % edgename
            # hyperedge head
            result += '%s -> n%s [label="%s"]\n' % \
                    (edgename,
                     edge.head.id_str(),
                     escape_quote(str(edge)) if detailed else '')
            # hyperedge tails
            for tailnode in edge.tail:
                result += 'n%s -> %s [dir="none"]\n' % \
                        (tailnode.id_str(), edgename)
        return result

    def neighbors(self, max_dist=3):
        """return a set of nodes who are within max_dist of self"""
        # TODO: this may have problems because the set doesn't
        # compare object id but uses user defined comparison methods
        # TODO: outgoing edges are no longer saved
        found = set()
        found.add(self)
        queue = [(self, 0)]
        while queue:
            node, d = queue.pop(0)
            if d < max_dist:
                for edge in node.outgoing:
                    if edge.head not in found:
                        found.add(edge.head)
                        queue.append((edge.head, d+1))
                for edge in node.incoming:
                    for tailnode in edge.tail:
                        if tailnode not in found:
                            found.add(tailnode)
                            queue.append((tailnode, d+1))
        return found

    def serialize(self):
        return '[%s]' % self.id

    def deserialize(self, s):
        s = s.strip()
        assert s.startswith('[') and s.endswith(']')
        self.id = int(s[1:-1])

    def show_neighborhood(self, max_dist=3, detailed=True):
        """show the neighborhood of this node in a picture"""
        dotstr = ''
        for node in self.neighbors(max_dist):
            if node is self:
                dotstr += node.dot(color='dodgerblue', detailed=detailed)
            else:
                dotstr += node.dot(detailed=detailed)
        dotstr = 'digraph hypergraph {\nrankdir=BT\n%s}\n' % dotstr
        f = open('/tmp/dotty', 'w')
        f.write(dotstr)
        f.close()
        os.system('cat /tmp/dotty | dot -Tgif > /tmp/dotty.gif')
        os.system('eog /tmp/dotty.gif')

class Edge(object):
    def __init__(self):
        # TODO: consider removing this head field to break the reference cycle
        # between a edge and its head. The reference cycle may cause memory
        # problems or an overhead on garbage collector
        self.head = None
        self.tail = []
        self.weight = 1
        self.hg = None  # hypergraph this edge belongs to

    def degree(self):
        return len(self.tail)

    def add_tail(self, node):
        self.tail.append(node)

    def posterior(self):
        """\sum{d: e \in d} w(d)"""
        prod = self.hg.prod
        w = self.hg.w
        p = self.head.outside
        for node in self.tail:
            p = prod(p, node.inside)
        p = prod(p, w(self))
        return p

    def expectation(self):
        """\sum{d: e \in d} w(d)f(d)"""
        zero = self.hg.zero
        one = self.hg.one
        sum = self.hg.sum
        prod = self.hg.prod
        w = self.hg.w
        f = self.hg.f

        term1 = self.head.outside
        for node in self.tail:
            term1 = prod(term1, node.inside)
        term1 = prod(term1, f(self))

        term2 = zero
        for v in self.tail:
            tmp = v.inside_exp
            for u in self.tail:
                if u is not v:
                    tmp = prod(tmp, u.inside)
            term2 = sum(term2, tmp)
        term2 = prod(term2, self.head.outside)

        term3 = self.head.outside_exp
        for node in self.tail:
            term3 = prod(term3, node.inside)

        return prod(w(self),
                    sum(term1, sum(term2, term3)))

    def make_path(self, subpaths):
        """subpaths is a list of paths on tail nodes.
        return a new path generated by concatenating this edge.
        this is used in k-best paths generation.

        boundary case (dangling edges under a node):
        if this edge has no tail node, make_path([]) returns a Path object
        with edge weight as its weight"""
        assert len(self.tail) == len(subpaths), '%s' % self
        path = Path(self, subpaths)
        weight = self.hg.one
        for p in subpaths:
            if p is not None:
                weight = self.hg.prod(weight, p.weight)
        weight = self.hg.prod(weight, self.hg.w(self))
        path.weight = weight
        return path

    def serialize(self):
        tail_str = ' '.join(tailnode.serialize() for tailnode in self.tail)
        head_str = self.head.serialize()
        return ' -> '.join([tail_str, head_str])

    def dot_label(self):
        return str(self)

    def deserialize(self, s):
        tail_nodes, head_node = s.split('->')
        tail_ids = [int(t[1:-1]) for t in tail_nodes.split()]
        head_id = int(head_node.strip()[1:-1])
        return tail_ids, head_id

class Path(object):
    """A path points to a edge and has a set of subpaths. Any subpaths can be
    None. e.g. A leaf node has None as its single path. This is also useful if
    a path does not start from the leaf nodes (treelets).

    If only complete paths are used, a complete path starts from either a leaf
    node (None path), or a dangling edge (a path object with empty subpaths
    list)."""
    def __init__(self, edge, subpaths):
        self.edge = edge
        self.subpaths = subpaths
        self.weight = 0

    def __lt__(self, other):
        best = self.edge.hg.one
        best = -10000
        if FLAGS.minimize_path_cost:
            return self.weight < other.weight
        else:
            return abs(best - self.weight) < abs(best - other.weight)

    def tree_str(self, indent=0):
        indent_level = 2
        myindent = ' '*indent
        result = '%s%s\n' % (myindent, self.edge.head)
        result += '%s%s\n' %  (myindent, self.edge)
        for p in self.subpaths:
            if p is None:
                result += '%s%s\n' % (myindent + ' '*indent_level, 'None')
            else:
                result += p.tree_str(indent + indent_level)
        return result

    def edges(self):
        yield self.edge
        for p in self.subpaths:
            if p is not None:
                for edge in p.edges():
                    yield edge

# semirings
LOGPROB = (logsum,
           logprod,
           LOGZERO,
           0)

INSIDE = (lambda x,y: x + y,
          lambda x,y: x*y,
          0,
          1)

SHORTEST_PATH = (lambda x,y: min([x,y]),
                 lambda x,y: x + y,
                 INF,
                 0)

class Hypergraph(object):
    def __init__(self, root):
        self.root = root
        # list of nodes in reverse topo order
        self.nodes = []
        # semiring operators
        #self.sum = lambda x,y: x + y   #Recent commented
        #self.prod = lambda x,y: x*y
        self.zero = 0
        self.one = 1
        #self.set_semiring(INSIDE)  #Recent commented
        # string flags of finished tasks
        self.tasks_done = set()

    def __iter__(self):
        for node in self.nodes:
            yield node

    def edges(self):
        for node in self:
            for edge in node.incoming:
                yield edge

    def set_semiring(self, semiring):
        self.sum, self.prod, self.zero, self.one = semiring

    def set_functions(self, w, f, g):
        self.w = w # weight function that defines prob distribution
        self.f = f # feature function 1
        self.g = g # feature function 2

    def find_reachable_nodes(self):
        """find nodes that are reachable from the top and count the
        number of outgoing edges for each node"""
        # find all reachable nodes down from the goal
        found = {}
        found[id(self.root)] = self.root
        queue = [self.root]
        #print >>sys.stderr, '---'
        while queue:
            node = queue.pop(0)
            if hasattr(node, 'dead'):
                if node.dead:
                    #print >>sys.stderr, 'dead', node
                    continue
                assert not node.dead
            for edge in node.incoming:
                for tailnode in edge.tail:
                    #print >>sys.stderr, tailnode
                    if id(tailnode) not in found:
                        found[id(tailnode)] = tailnode
                        queue.append(tailnode)
                        tailnode.nout = 0
                    tailnode.nout += 1
        # save for sanity check
        self.found = found

    def topo_sort(self):
        """top down topo sort. nodes that don't reach the target node are thrown
        away"""
        # TODO: detect cycles
        self.find_reachable_nodes()
        # save list of nodes in topo order
        self.nodes = []
        # assign each node an id field incrementally
        cur_id = 0
        # count visited outgoing edges for each node
        unvisited = {}
        for nid, node in self.found.items():
            unvisited[nid] = node.nout
        queue = [self.root]
        #print >>sys.stderr, '+++'
        while queue:
            # take off nodes whose all outgoing edges are visited from
            # queue head
            node = queue.pop(0)
            self.nodes.append(node)
            node.hg = self
            node.id = cur_id
            cur_id += 1
            for edge in node.incoming:
                edge.hg = self
                for tailnode in edge.tail:
                    #print >>sys.stderr, tailnode
                    unvisited[id(tailnode)] -= 1
                    if unvisited[id(tailnode)] == 0:
                        queue.append(tailnode)
        self.sanity_check()
        self.tasks_done.add('topo_sort')

    def sanity_check(self):
        sorted_all_reachable_nodes = (len(self.found) == len(self.nodes))
        if not sorted_all_reachable_nodes:
            logger.writeln('reachable nodes: %s' % len(self.found))
            logger.writeln('sorted nodes: %s' % len(self.nodes))
            logger.writeln('nodes found but not sorted:')
            nids = [id(node) for node in self.nodes]
            for nid, node in self.found.items():
                if nid not in nids:
                    logger.writeln()
                    logger.writeln(node)
                    logger.writeln('--edges--')
                    for edge in node.incoming:
                        logger.writeln(edge)
                    # for decoder hypergraph only
                    if hasattr(node, 'unary_chain'):
                        logger.writeln('--unary chain--')
                        for n in node.unary_chain:
                            logger.writeln(n)
        assert sorted_all_reachable_nodes, 'cycles may exist in graph'
        del self.found

    def assert_done(self, task):
        """make sure a given operation on the hypergraph is done"""
        if task not in self.tasks_done:
            # do the task if not done
            method = getattr(self, task)
            method()

    def topo_order(self):
        """bottom-up topological order"""
        self.assert_done('topo_sort')
        for node in reversed(self.nodes):
            yield node

    def reverse_topo_order(self):
        """top-down topological order"""
        self.assert_done('topo_sort')
        for node in self.nodes:
            yield node

    def inside(self):
        for node in self.topo_order():
            # initialization
            if not node.incoming:
                node.inside = self.one
                continue
            score = self.zero
            for edge in node.incoming:
                edge_score = self.w(edge)
                for tailnode in edge.tail:
                    edge_score = self.prod(edge_score, tailnode.inside)
                score = self.sum(score, edge_score)
            node.inside = score
        self.tasks_done.add('inside')

    def outside(self):
        self.assert_done('inside')
        # initialization
        for node in self.reverse_topo_order():
            if node is self.root:
                node.outside = self.one
            else:
                node.outside = self.zero
        for node in self.reverse_topo_order():
            for edge in node.incoming:
                score = self.prod(self.w(edge), node.outside)
                for tailnode in edge.tail:
                    tmp = score
                    for othernode in edge.tail:
                        if othernode is not tailnode:
                            tmp = self.prod(tmp, othernode.inside)
                    tailnode.outside = self.sum(tailnode.outside, tmp)
        self.tasks_done.add('outside')

    def inside_exp(self):
        self.assert_done('inside')
        for node in self.topo_order():
            node.inside_exp = self.zero
            for edge in node.incoming:
                term1 = self.f(edge)
                for v in edge.tail:
                    term1 = self.prod(term1, v.inside)

                term2 = self.zero
                for v in edge.tail:
                    tmp = v.inside_exp
                    for w in edge.tail:
                        if w is not v:
                            tmp = self.prod(tmp, w.inside)
                    term2 = self.sum(term2, tmp)

                term = self.prod(self.w(edge), self.sum(term1, term2))
                node.inside_exp = self.sum(node.inside_exp, term)
        self.tasks_done.add('inside_exp')

    def outside_exp(self):
        self.assert_done('inside_exp')
        for node in self.reverse_topo_order():
            node.outside_exp = self.zero
        for node in self.reverse_topo_order():
            for edge in node.incoming:
                for tailnode in edge.tail:
                    term1 = self.f(edge)
                    term1 = self.prod(term1, node.outside)
                    for v in edge.tail:
                        if v is not tailnode:
                            term1 = self.prod(term1, v.inside)

                    term2 = self.zero
                    for v in edge.tail:
                        if v is not tailnode:
                            tmp = v.inside_exp
                            for w in edge.tail:
                                if w is not tailnode and w is not v:
                                    tmp = self.prod(tmp, w.inside)
                            term2 = self.sum(term2, tmp)
                    term2 = self.prod(term2, node.outside)

                    term3 = node.outside_exp
                    for v in edge.tail:
                        if v is not tailnode:
                            term3 = self.prod(term3, v.inside)

                    term = self.prod(self.w(edge),
                                     self.sum(term1, self.sum(term2, term3)))
                    tailnode.outside_exp = self.sum(tailnode.outside_exp, term)
        self.tasks_done.add('outside_exp')

    def dot(self, detailed=True):
        """dot language representation"""
        self.assert_done('topo_sort')
        result = ''
        # write hypernodes
        for node in self.topo_order():
            result += node.dot(detailed=detailed)
        return 'digraph hypergraph {\nrankdir=BT\n%s}\n' % result

    def show(self):
        """show the hypergraph as a pic"""
        f = open('/tmp/dotty', 'w')
        f.write(self.dot())
        f.close()
        os.system('cat /tmp/dotty | dot -Tgif > /tmp/dotty.gif')
        os.system('eog /tmp/dotty.gif')

    def serialize(self, filename):
        """write to file"""
        f = open(filename, 'w')
        for node in self:
            f.write('%s\n' % node.serialize())
            for edge in node.incoming:
                f.write('%s\n' % edge.serialize())
        f.close()

    def stats(self):
        header = '{:-^50}\n'
        field = '{:<35}{:>15}\n'
        result = header.format('Hypergraph Stats')
        self.assert_done('topo_sort')
        result += field.format('[nodes]:',
                               len(self.nodes))
        result += field.format('[edges]:',
                               sum(len(node.incoming) for node in self.nodes))
        return result

class Deserializer(object):
    def __init__(self, node_class=Node, edge_class=Edge):
        """A deserializer reads hypergraphs dumped into files back into memory.
        The deserializer is initialized with types of nodes and edges used to
        construct a hypergraph."""
        self.node_class = node_class
        self.edge_class = edge_class

    def deserialize(self, filename):
        """Read a file and return a toposorted hypergraph."""
        f = open(filename)
        edges_tails = []
        nodes = []
        # first pass adds incoming edges to nodes
        for line in f:
            if '->' in line:  # edge
                edge = self.edge_class()
                tail_ids, head_id = edge.deserialize(line)
                nodes[head_id].add_incoming(edge)
                edges_tails.append((edge, tail_ids))
            else:  # node
                node = self.node_class()
                node.deserialize(line)
                assert node.id == len(nodes), 'nodes shall appear in order'
                nodes.append(node)
        # second pass adds tail nodes to edges
        for edge, tail_ids in edges_tails:
            for nid in tail_ids:
                edge.add_tail(nodes[nid])
        f.close()
        # make a toposorted hypergraph
        hg = Hypergraph(nodes[0])
        hg.nodes = nodes
        for node in hg:
            node.hg = hg
        for edge in hg.edges():
            edge.hg = hg
        hg.tasks_done.add('topo_sort')
        return hg
