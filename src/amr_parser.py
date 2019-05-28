#!/usr/bin/env python3
import sys
from collections import defaultdict
def surface_check_var(s):
    if s[0] < 'a' or s[0] >'z':
        return False
    if len(s) == 1:
        return True
    for i in range(1, len(s)):
        if s[i] < '0' or s[i] > '9':
            return False
    return True

def from_AMR_line(line):
    fragments = []
    state=-1 #significant symbol just encountered: 1 for (, 2 for :, 3 for /
    stack=[] #variable stack
    cur_charseq=[] #current processing char sequence
    var_dict={} #key: var name value: var value
    var_list=[] #variable name list (order: occurence of the variable
    var_attr_dict1=defaultdict(list) #key: var name:  value: list of (attribute name, other variable)
    #var_attr_dict2=defaultdict(list) #key:var name, value: list of (attribute name, const value)
    cur_attr_name="" #current attribute name
    attr_list=[] #each entry is an attr dict
    in_quote=False
    for i,c in enumerate(line.strip()):
        if c==" ":
            if state==2:
                cur_charseq.append(c) #will be stripped
            continue
        if c=="\"": #if it's the first quote or the second
            if in_quote:
                in_quote=False
            else:
                in_quote=True
        if c=="(":
            if in_quote:
                continue
            if state==2:
                if cur_attr_name!="":
                    print("Format error when processing ",line[0:i+1], file=sys.stderr)
                    return None
                cur_attr_name="".join(cur_charseq).strip() #just identified a relation before a concept after parenthesis
                cur_charseq[:]=[]
            state=1
        elif c==":":
            if in_quote:
                continue
            if state==3: #(...: #/  Has just identified a variable value
                var_value="".join(cur_charseq)
                cur_charseq[:]=[]
                cur_var_name=stack[-1] #The variable will be pushed into stack before recognizing its value
                var_dict[cur_var_name]=var_value
            elif state==2: #: ...:  What is this situation, only when const, we have recognized relation and the const it links to, or it is a variable
                temp_attr_value="".join(cur_charseq)
                cur_charseq[:]=[]
                parts=temp_attr_value.split()
                if len(parts)<2:
                    print("Error in processing",line[0:i+1], file=sys.stderr)
                    return None
                attr_name=parts[0].strip()
                attr_value=(' '.join(parts[1:])).strip()
                if len(stack)==0:
                    print("Error in processing",line[:i],attr_name,attr_value, file=sys.stderr)
                    return None
                if attr_value not in var_dict:
                    if surface_check_var(attr_value): #The first appearance of the variable
                        var_attr_dict1[stack[-1]].append((attr_name,attr_value, True, True)) #var_attr_dict2 recognize relation between a variable and a const
                    else:
                        var_attr_dict1[stack[-1]].append((attr_name,attr_value, False, False)) #var_attr_dict2 recognize relation between a variable and a const
                else:
                    var_attr_dict1[stack[-1]].append((attr_name,attr_value, True, True)) #dict1 recognize relation between a variable and a variable
            state=2
        elif c=="/":
            if in_quote:
                #try:
                #    assert False, 'This should not happen, it is now in %s' % line
                #except:
                #    print >> sys.stderr, 'This should not happen, it is now in %s' % line
                continue
            if state==1:
                variable_name="".join(cur_charseq)
                cur_charseq[:]=[]
                if variable_name in var_dict:
                    print("Duplicate variable ",variable_name, " in parsing AMR", file=sys.stderr)
                    return None
                stack.append(variable_name)
                var_list.append(variable_name)
                if cur_attr_name!="": #There are unassigned relation
                    if not cur_attr_name.endswith("-of"): #Transformation of -of relation
             # var_attr_dict1[stack[-2]][cur_attr_name]=variable_name
                        var_attr_dict1[stack[-2]].append((cur_attr_name, variable_name, True, False))
                    else:
                        var_attr_dict1[stack[-2]].append((cur_attr_name, variable_name, True, False))
                    cur_attr_name=""
            else:
                print("Error in parsing AMR", line[0:i+1], file=sys.stderr)
                return None
            state=3
        elif c==")":
            if in_quote:
                continue
            if len(stack)==0:
                print("Unmatched parathesis at position", i, "in processing", line[0:i+1], file=sys.stderr)
                return None
            if state==2:
                temp_attr_value="".join(cur_charseq)
                cur_charseq[:]=[]
                parts=temp_attr_value.split()
                if len(parts)<2:
                    print("Error processing",line[:i+1],temp_attr_value, file=sys.stderr)
                    return None
                attr_name=parts[0].strip()
                attr_value=(' '.join(parts[1:])).strip()
                if cur_attr_name.endswith("-of"): #What is this situation?
                    var_attr_dict1[stack[-2]].append((cur_attr_name, variable_name, True, False))
                elif attr_value not in var_dict:
                    if surface_check_var(attr_value):
                        var_attr_dict1[stack[-1]].append((attr_name,attr_value, True, True))
                    else:
                        var_attr_dict1[stack[-1]].append((attr_name,attr_value, False, False))
                else:
                    var_attr_dict1[stack[-1]].append((attr_name,attr_value, True, True))
            elif state==3:
                var_value="".join(cur_charseq)
                cur_charseq[:]=[]
                cur_var_name=stack[-1]
                var_dict[cur_var_name]=var_value
            stack.pop()  #Have met the sign of recognizing a whole variable
            cur_attr_name=""
            state=4 #just mark as not doing any relations when metting other state symbols
        else:
            cur_charseq.append(c)

    var_value_list=[] #Keep a list of all the values for all the variables
    link_list=[]
    const_attr_list=[]

    for v in var_list:
        if v not in var_dict:
            print("Error: variable value not found", v, file=sys.stderr)
            return None
        else:
            var_value_list.append(var_dict[v])

    return (var_list, var_value_list, var_attr_dict1)
