# Allison Fellger and Angela Rae
# 4.26.2019

# Creates a decision tree based on entropy in order to classify 
#   test scores

import math
import utils2 as utils
import utils as u
import numpy as np
import random

LEAF = 'leaf'
SPLIT = 'split'
UNSPLIT = 'unsplit'


class TreeNode(object):
    def __init__(self, table, header, first=True, full_table=None):
        self.branches = {}
        self.table = table
        self.node_type = None
        self.split_index = None
        self.leaf_class = None
        self.header = header

        if first:
            full_table = []
            for i, col in enumerate(header[:-1]):
                full_table.append(utils.unique([x[i] for x in table]))
        
        classes = [x[-1] for x in table]
        c_types = utils.unique(classes)
        
        if len(c_types) == 1:
            ut = utils.unique_table(self.table)
            self.node_type = LEAF
            self.leaf_class = c_types[0]

        else:
            self.split_index = max_gain(table, header)

            if self.split_index != -1:
                split_vals = utils.unique(table, col=self.split_index)
                self.node_type = SPLIT
                branch_tabs = [[y for y in table if y[self.split_index] == x] for x in split_vals]
                for i, bran in enumerate(branch_tabs):
                    self.branches[split_vals[i]] = TreeNode(bran, header, first=False, full_table=full_table)
            else:
                self.node_type = LEAF
                self.leaf_class = utils.majority_vote(table)
                ut = utils.unique_table(self.table)

    
    def classify(self, instance):
        '''
            Given an instance, return the leaf class that the tree
            classifies it as.
        '''
        if self.node_type == LEAF:
            return self.leaf_class
        else:
            new_att = instance[self.split_index]
            if new_att in self.branches:
                return self.branches[new_att].classify(instance)
            else:
                return utils.majority_vote(self.table)
#------------------------------------------------------
#   Step 1: Interview Classifier
#------------------------------------------------------

def entropy(table):
    '''
        Calculate the entropy of a set of instances.
    '''
    e = 0
    classes = [x[-1] for x in table]
    c_types = utils.unique(classes)
    for c in c_types:
        c_ratio = sum([1 for x in classes if x == c])/len(classes)
        if c_ratio != 0:
            e += c_ratio * math.log(c_ratio, 2)
    return -e

def info_gain(table, att_i):
    '''
        calculate informaiton gain for one attribute
    '''
    e_start = entropy(table)
    e_new = 0
    atts = utils.unique([x[att_i] for x in table])
    t_size = len(table)
    for a in atts:
        partition = [x for x in table if x[att_i] == a]
        p_weight = len(partition)/t_size
        e_new += (entropy(partition) * p_weight)
    return e_start - e_new

def max_gain(table, header):
    '''
        calculate all possible information gains and return the index of the
        instance that will maximize it
    '''
    i_gains = {}
    for i, col in enumerate(header[:-1]):
        if not utils.unanimous(table, i):
            i_gains[info_gain(table, i)] = i
    if len(i_gains) != 0:
        return i_gains[max(i_gains)]
    else:
        return -1

def s_err(tp, tn, fp, fn):
    '''
        calculate standard error
    '''
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    err = math.sqrt((correct/ttl * incorrect/ttl) / ttl)
    return err

def acc(tp, tn, fp, fn):
    '''
        calculate accuracy (correct classifications / all classifications)
    '''
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    return correct/ttl

def test_tree(header, training_table, test_table):
    '''
        given training and test sets, generate a decision tree and
        return a tuple containing the count of true positives, true
        negatives, false positives and false negatives.
    '''
    model = TreeNode(training_table, header)
    con_mat = [[0 for x in range(4)] for y in range(4)]
    
    for inst in test_table:
        p = model.classify(inst) - 1
        a = inst[-1] - 1
        if p != -1:
            con_mat[a][p] += 1
        
    return con_mat
    

def c_matrix(tp, tn, fp, fn):  
    '''
        Prints a formatted confusion matrix.
    '''
    print('\n                        Predicted       ')  
    print('         |-------------------------------------|')
    print('         |       |    Yes  |    No   |  Total  |')  
    print('         |-------------------------------------|')
    print('         |   Yes |  %5d  |  %5d  |  %5d  |' % (tp, fn, tp+fn))
    print('  Actual |-------------------------------------|')
    print('         |   No  |  %5d  |  %5d  |  %5d  |' % (fp, tn, fp+tn))
    print('         |-------------------------------------|')
    print('         | Total |  %5d  |  %5d  |  %5d  |' % (tp+fp, fn+tn, tp+tn+fp+fn))
    print('         |-------------------------------------|\n')
    print('              Accuracy     : %.5f' % acc(tp, tn, fp, fn))
    print('              Standard Err : %.5f\n' % s_err(tp, tn, fp, fn))

def k_cross(header, table, k):
    '''
        Divides table into k groups. Then, for each fold, use all other
        groups together as a training set to create a tree to classify instances
        in the fold. Prints a confusion matrix of the resulting accuracy.
    '''
    random.shuffle(table)
    i = 0
    folds = [[] for x in range(k)]

    stratified = [a for b in [[x for x in table if x[-1] == y] for y in range(1, 5)] for a in b]
    j = 0

    for s in stratified:
        folds[j].append(s)
        j = (j + 1) % k

    con_mat = [[0 for x in range(4)] for y in range(4)]
    for i in folds:
        test = i
        train = [x for y in folds for x in y if y != test]
        temp_mat = test_tree(header, train, test)
        con_mat = utils.add_tables(con_mat, temp_mat)

    raw_c_mat(con_mat)
    get_mat_accuracy(con_mat)

def raw_c_mat(outcome):
    n = len(outcome)
    for i in range(n+1):
        print('%4d' % i, end='')
    print()
    
    for i in range(n):
        print('%4d' % (i + 1), end='')
        for j in range(n):
            print('%4d' % outcome[i][j], end='')
        print()

def get_mat_accuracy(con_mat):
    total = sum([sum(x) for x in con_mat])

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for i in range(len(con_mat)):
        newtp, newtn, newfp, newfn = one_mpg_acc(i, con_mat, total)
        tp += newtp
        tn += newtn
        fp += newfp
        fn += newfn
    
    c_matrix(tp, tn, fp, fn)

def one_mpg_acc(i, con_mat, total):
    tp = con_mat[i][i]
    fp = sum(con_mat[i]) - tp
    fn = sum([x[i] for x in con_mat]) - tp
    tn = total - (tp + fp + fn)
    return tp, tn, fp, fn

def group_scores(students):
    outlist = []
    rawscores = [sum([int(x.strip('"')) for x in s[-3:]]) for s in students]
    q1 = np.quantile(rawscores, 0.25)
    q2 = np.quantile(rawscores, 0.50)
    q3 = np.quantile(rawscores, 0.75)

    for s in students:
        outlist.append([x.strip('"') for x in s[:-3]])
        score = sum([int(x.strip('"')) for x in s[-3:]])
        if score < q1:
            c = 1
        elif score < q2:
            c = 2
        elif score < q3:
            c = 3
        else:
            c = 4
        outlist[-1].append(c)
    return outlist

def main():
    tname = 'StudentsPerformance.csv'
    students = u.read_table(tname)
    header = students[0][:-3] + ['AvgScore']
    students = group_scores(students[1:])
    # t_head, t_tab = clean_titanic(tname)
    out = k_cross(header, students, 10)
    # print("Mean AbsoluteError: ", out)
main()
