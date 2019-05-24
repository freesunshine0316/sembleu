#!/usr/bin/python
import sys, re
months = {'January':1, 'February':2, 'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 'August':8, 'September':9, 'October':10, 'November':11, 'December':12, 'Sept.':9}
money_quantity = {'thousand':1000, 'million':1000000, 'billion':1000000000, 'trillion':1000000000000}
timezone = set(['UTC', 'GMT', 'AEDT', 'EST', 'DST'])
days = set(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
quantities = {'one':1, 'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10, 'hundred':100, 'hundreds':100, 'thousand':1000, 'thousands':1000, 'millions':1000000, 'million':1000000, 'billion':1000000000, 'trillion':1000000000000}
def is_num(s):
    regex = re.compile('[0-9]+([^0-9\s]*)')
    match = regex.match(s)
    return match and len(match.group(1)) == 0

def is_percentage(s):
    if is_num(s):
        return True
    regex = re.compile('[0-9]+\.[0-9]+([^0-9]*)')
    match = regex.match(s)
    return match and len(match.group(1)) == 0

def is_year(tok):
    return is_num(tok) and len(tok) == 4 and int(tok[0]) < 4

def is_day(tok):
    if not is_num(tok):
        return False
    date = int(tok)
    return date >= 0 and date <= 31

def date_extraction(toks):
    if len(toks) == 4:
        if toks[0] in months and is_num(toks[1]) and toks[2] == ',' and is_num(toks[3]):
            return (True, '[A1-1]', '(. :d/date-entity :year (. :%d ) :month (. :%d ) :day (. :%d ))' % (int(toks[3]), months[toks[0]], int(toks[1])))

    if len(toks) == 5:
        if is_year(toks[0]) and toks[1] == '@-@' and is_num(toks[2]) and toks[3] == '@-@' and is_num(toks[4]):
            return (True, '[A1-1]', '(. :d/date-entity :year (. :%d ) :month (. :%d ) :day (. :%d ))' % (int(toks[0]), int(toks[2]), int(toks[4])))

    if len(toks) == 5:
        if is_num(toks[0]) and toks[1] == '@:@' and is_num(toks[2]) and toks[3] == '@:@' and is_num(toks[4]):
            return (True, '[A1-0]', '(. :time (. :\"%s%s%s\" ))' % (toks[0], toks[2], toks[4]))

    if len(toks) == 2:
        if toks[0] in months and is_year(toks[1]):
            return (True, '[A1-1]', '(. :d/date-entity :year (. :%d ) :month (. :%d ))' % (int(toks[1]), months[toks[0]]))

    if len(toks) == 3:
        if toks[0] in months and is_day(toks[1]) and toks[2] == 'th':
            return (True, '[A1-1]', '(. :d/date-entity :month (. :%d ) :day (. :%d ))' % (months[toks[0]], int(toks[1])))

    if len(toks) == 1:
        if is_year(toks[0]):
            return (True, '[A1-1]', '(. :d/date-entity :year (. :%d ))' % (int(toks[0])))

    if len(toks) == 1:
        curr_tok = toks[0]
        fields = curr_tok.split('.')
        if len(fields) == 3:
            all_num = True
            for t in fields:
                if not is_num(t):
                    all_num = False
                    break
            if all_num:
                if ((len(fields[0]) == 3) and (len(fields[1])==3) and (len(fields[2])==4)):
                    return (True, '[A1-1]', '(. :p/phone-number-entity :value (. :%s ))' % (''.join(fields)))

    if len(toks) == 5:
        tmp = toks[4]
        if not is_num(toks[4]):
            tmp = toks[4][:-1]

        if is_num(toks[0]) and is_num(toks[2]) and is_num(tmp) and toks[1] == '@-@' and toks[3] == '@-@' and len(toks[0]) == 3 and len(toks[2]) == 3 and len(tmp) == 4:
            return (True, '[A1-1]', '(. :p/phone-number-entity :value (. :%s ))' % (toks[0]+toks[2]+tmp))

    if len(toks) == 1:
        if '@' in toks[0] and ('.com' in toks[0] or '.org' in toks[0]):
            return (True, '[A1-1]', '(. :e/email-address-entity :value (. :%s ))' % toks[0])

    if len(toks) == 1:
        if toks[0] in timezone:
            return (True, '[A1-0]', '(. :timezone (. :"%s" ))' % toks[0])

    if len(toks) == 1:
        if toks[0][0] == '@' and toks[0][-1] != '@': #This is at someone
            return (True, '[A1-1]', '(. :p/person :name (. :n/name :op1 (. :"%s")) :wiki (. :-))' % toks[0][1:])

    #Then percentage entity
    if len(toks) == 2:
        if is_percentage(toks[0]) and toks[1] == '%':
            percentage_str = '(. :p/percentage-entity :value (. :%s ))' % toks[0]
            return (True, '[A1-1]', percentage_str)

    #monetary quantity
    if len(toks) == 3:
        if toks[0] == '$' and is_percentage(toks[1]) and toks[2] in money_quantity:
            amount = money_quantity[toks[2]] * float(toks[1])
            amount = '%d' % int(amount)

            money_str = '(. :m/monetary-quantity :quant (. :%s ) :unit (. :d/dollar ))' % amount
            return (True, '[A1-1]', money_str)

    #if len(toks) == 1:
    if len(toks) == 3:
        is_quantity = True
        amount = 1
        for i in xrange(3):
            if is_percentage(toks[i]) or toks[i] in quantities:
                if toks[i] in quantities:
                    amount *= float(quantities[toks[i]])
                else:
                    amount *= float(toks[i])
            else:
                is_quantity = False
                break

        if is_quantity:
            amount = int(amount)
            quantity_str = '(. :quant (. :%d ))' % amount
            return (True, '[A1-0]', quantity_str)

    if len(toks) == 2:
        is_quantity = True
        amount = 1
        for i in xrange(2):
            if is_percentage(toks[i]) or toks[i] in quantities:
                if toks[i] in quantities:
                    amount *= float(quantities[toks[i]])
                else:
                    amount *= float(toks[i])
            else:
                is_quantity = False
                break
        if is_quantity:
            amount = int(amount)
            quantity_str = '(. :quant (. :%d ))' % amount
            return (True, '[A1-0]', quantity_str)

    if len(toks) == 1:
        if toks[0] in days:
            return (True, '[A1-0]', '(. :weekday (. :%s/%s ))' % (toks[0][0].lower(), toks[0].lower()))

    return (False, None, None)

def extract_all_dates(file):
    all_dates = []
    f = open(file, 'r')
    for (i, line) in enumerate(f):
        #print line.strip()
        dates_in_line = []
        line = line.strip()
        if line:
            toks = line.split()
            n_toks = len(toks)
            aligned = set()
            for start in xrange(n_toks):
                if start in aligned:
                    continue
                for length in xrange(n_toks+1, 0, -1):
                    end = start + length
                    if end > n_toks:
                        continue

                    span_set = set(xrange(start, end))
                    if len(span_set & aligned) != 0:
                        continue

                    (is_date, lhs, rule_str) = date_extraction(toks[start:end])
                    if is_date:
                        #print line
                        #print '%d-%d %s' % (start, end, toks[start:end])
                        #print '%s ||| %s' % (lhs, rule_str)
                        dates_in_line.append((start, end, lhs, rule_str))

                        aligned |= span_set
                        break
            all_dates.append(dates_in_line)
    f.close()
    return all_dates

if __name__ == '__main__':
    extract_all_dates(sys.argv[1])
