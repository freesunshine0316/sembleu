import sys
import datetime

import gflags
FLAGS = gflags.FLAGS

gflags.DEFINE_boolean(
    'time_stamp',
    False,
    'Print time stamps.')

level = 1
file = sys.stderr

def writeln(s=""):
    if FLAGS.time_stamp:
        file.write('[%s] %s\n' % (datetime.datetime.utcnow(), s))
    else:
        file.write("%s\n" % s)
    file.flush()

def write(s):
    file.write(s)
    file.flush()
