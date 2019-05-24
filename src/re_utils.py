import re
def extract_patterns(s_repr, pattern):
    #print s_repr
    #print pattern
    re_pat = re.compile(pattern)
    position = 0
    matched_list = []
    while position < len(s_repr):
        match = re_pat.search(s_repr, position)
        if not match:
            break
        matched_s = match.group(0)
        matched_list.append(matched_s)
        #print matched_s
        position = match.end()
    return matched_list

def delete_pattern(s_repr, pattern):
    re_pat = re.compile(pattern)
    position = 0
    matched_list = []
    while position < len(s_repr):
        match = re_pat.search(s_repr, position)
        if not match:
            break
        matched_s = match.group(0)
        matched_list.append(matched_s)
        position = match.end()

    for matched in matched_list:
        s_repr = s_repr.replace(matched, '')
    return s_repr

def parse_indexes(toks):
    toks = [tok.split('.')[1] for tok in toks]
    spans = []
    for tok in toks:
        spans += [int(p) for p in tok.split(',')]
    return spans

def extract_entity_spans(frag, opt_toks, role_toks, unaligned):
    op_indexs = []
    role_index_set = set()
    #First find all ops that are aligned to this frag
    for (s_index, e_index) in opt_toks:
        if frag.edges[e_index] == 1:
            op_indexs.append(s_index)

    for (s_index, e_index) in role_toks:
        if frag.edges[e_index] == 1:
            role_index_set.add(s_index)

    if len(op_indexs) == 0:
        return (None, None, None)

    op_indexs = sorted(op_indexs)
    op_set = set(op_indexs)
    start = op_indexs[0]
    end = start + 1
    m_end = start + 1
    while True:
        if end in op_set:
            end += 1
            m_end = end
        elif end in role_index_set:
            end += 1
            m_end = end
        elif end in unaligned:
            end += 1
        else:
            break
    return (start, m_end, m_end < op_indexs[-1])

    #toks = extract_patterns(frag_repr, '~e\.[0-9]+(,[0-9]+)*')
    #toks = [tok.split('.')[1] for tok in toks]
    #spans = []
    #for tok in toks:
    #    spans += [int(p) for p in tok.split(',')]

    #if len(spans) == 0:
    #    return None
    #spans.sort()
    #continous_spans = []
    #start = spans[0]
    #end = start + 1
    #for tok in spans[1:]:
    #    if tok == end - 1:
    #        continue
    #    if tok == end:
    #        end += 1
    #    else:
    #        continous_spans.append((start, end))
    #        start = tok
    #        end = tok + 1
    #if len(continous_spans) == 0 or continous_spans[-1] != (start, end):
    #    continous_spans.append((start, end))
    #return continous_spans

if __name__ == '__main__':
    #print extract_patterns('. :t/tour~e.8 :wiki (. :-) :name (. :n/name :op1 (. :"APOLOGIES"~e.5) :op2 (. :"ON"~e.6) :op3 (. :"BEER"~e.7,10))', '~e\.[0-9]+(,[0-9]+)*')
    print extract_entity_spans('. :t/tour~e.8 :wiki (. :-) :name (. :n/name :op1 (. :"APOLOGIES"~e.5) :op2 (. :"ON"~e.6) :op3 (. :"BEER"~e.7,10))')
