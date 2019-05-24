#!/usr/bin/python
import sys
def filter_vars(line):
    fields = line.split('|||')
    parts = fields[2].split('/')
    for i in xrange(len(parts)):
        try:
            while parts[i][-1] in '0123456789':
                parts[i] = parts[i][:-1]
        except:
            print line
            print parts
    fields[2] = '/'.join(parts)
    return '|||'.join(fields)

def main(argv):
    file = argv[1]
    stop_words_file = argv[2]
    result_file = argv[3]
    wf = open(result_file, 'w')
    stop_words = set()

    for line in open(stop_words_file, 'r'):
        line = line.strip()
        if line != '':
            stop_words.add(line)
            line = line[0].upper() + line[1:]
            stop_words.add(line)
            line = line.upper()
            stop_words.add(line)

    stop_set = set()
    for s in stop_words:
        stop_set.add('/%s ' % s)
        stop_set.add('/%s)' % s)
        stop_set.add('/%s )' % s)
    for line in open(file, 'r'):
        if line.strip() == '':
            wf.write(line)
            continue
        try:
            lex_part = line.strip().split('|||')[1].strip()
        except:
            print line
        graph_part = line.strip().split('|||')[2].strip()
        if lex_part not in stop_words:
            no_stop = True
            #for w in stop_set:
            #    if w in graph_part:
            #        no_stop = False
            #        break
            if no_stop:
                wf.write(filter_vars(line))
    wf.close()

if __name__ == '__main__':
    main(sys.argv)
