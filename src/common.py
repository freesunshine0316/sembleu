#!/usr/bin/env python
import sys
import time

import gflags
FLAGS = gflags.FLAGS
import logger

INF = 1e300
ZERO = 1e-100

gflags.DEFINE_integer(
    'time_interval',
    1000,
    'Output time result every #time_interval# items processed.')
gflags.DEFINE_string(
    'do',
    None,
    'A single number ("5") or a range ("5-10") of items to process.')

def count_lines(filename):
    f = open(filename)
    n = 0
    for line in f:
        n += 1
    return n

def cyk_spans(n):
    for span in range(1, n+1):
        for i in range(0, n-span+1):
            j = i + span
            yield (i, j)

def bi_cyk_spans(n1, n2):
    """Generate bilingual boxes in topological order.
    (f_span_begin, f_span_end, e_span_begin, e_span_end)"""
    for span1 in range(1, n1+1):
        for span2 in range(1, n2+1):
            for i1 in range(0, n1-span1+1):
                j1 = i1 + span1
                for i2 in range(0, n2-span2+1):
                    j2 = i2 + span2
                    yield (i1, j1, i2, j2)

def timed(l):
    prev = time.time()
    for i, x in enumerate(l, 1):
        if i % FLAGS.time_interval == 0:
            logger.writeln('%s (%s/sec)' %
                           (i, FLAGS.time_interval/(time.time()-prev)))
            prev = time.time()
        yield x

def select(l):
    start = 0
    end = INF
    if FLAGS.do:
        start = 0
        end = INF
        if '-' in FLAGS.do:
            a, b = FLAGS.do.split('-')
            try:
                start = int(a)
            except ValueError:
                pass
            try:
                end = int(b)
            except ValueError:
                pass
        else:
            start = end = int(FLAGS.do)
    for i, x in enumerate(l, 1):
        if i > end:
            break
        if start <= i <= end:
            yield x

def parse_flags():
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        logger.writeln('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)
    return argv

if __name__ == '__main__':
    import sys
    try:
        argv = FLAGS(sys.argv)  # parse flags
    except gflags.FlagsError as e:
        print('%s\nUsage: %s ARGS\n%s' % (e, sys.argv[0], FLAGS))
        sys.exit(1)

    for i in select(timed(range(1000))):
        print(i)
