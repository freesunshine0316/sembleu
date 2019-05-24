#!/usr/bin/env python3

def mark_level(hg):
    """node.level = the number of minimal rules on each hyperpath.
    also make sure this hg has the property that each hyperpath has the same
    number of edges"""
    for node in hg.topo_order():
        node.level = 0
        for edge in node.incoming:
            level = sum(c.level for c in edge.tail) + 1
            if node.level == 0:
                node.level = level
            else:
                assert node.level == level, '%s level %s or %s?' % \
                        (node, node.level, level)

