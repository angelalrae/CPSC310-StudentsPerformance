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

TP = 'true_pos'
TN = 'true_neg'
FP = 'false_pos'
FN = 'false_neg'


class TreeNode(object):
    def __init__(self, table, header):
        self.branches = {}
        self.table = table
        self.node_type = None
        self.split_index = None
        self.leaf_class = None
        self.header = header
        
        classes = [x[-1] for x in table]
        c_types = utils.unique(classes)
        
        if len(c_types) == 1:
            self.node_type = LEAF
            self.leaf_class = c_types[0]
            # print("Creating leaf: ")
            # print(self.table)
            # print(self.leaf_class)
        
        else:
            self.split_index = max_gain(table, header)
            if self.split_index != -1:
                split_vals = utils.unique(table, col=self.split_index)
                self.node_type = SPLIT
                branch_tabs = [[y for y in table if y[self.split_index] == x] for x in split_vals]
                for i, bran in enumerate(branch_tabs):
                    self.branches[split_vals[i]] = TreeNode(bran, header)
            else:
                self.node_type = LEAF
                self.leaf_class = utils.majority_vote(table)
                # print("Creating leaf: ")
                # print(self.table)
                # print(self.leaf_class)
    
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

def test_tree(header, training_table, test_table, result):
    '''
        given training and test sets, generate a decision tree and
        return a tuple containing the count of true positives, true
        negatives, false positives and false negatives.
    '''
    model = TreeNode(training_table, header)
    for inst in test_table:
        p = model.classify(inst)
        a = inst[-1]
        if p == a:
            if p == 1:
                result[TP] += 1
            else :
                result[TN] += 1
        else:
            if p == 1:
                result[FP] += 1
            else:
                result[FN] += 1


def k_cross(header, table, k):
    '''
        Divides table into k groups. Then, for each fold, use all other
        groups together as a training set to create a tree to classify instances
        in the fold. Prints a confusion matrix of the resulting accuracy.
    '''

    t = [x for x in table if x[-1] == 1]
    f = [x for x in table if x[-1] == 0]
    random.shuffle(t)
    random.shuffle(f)
    stratified = t + f

    folds = [[] for x in range(k)]
    i = 0

    for s in stratified:
        folds[i].append(s)
        i = (i + 1) % k

    
    results = {TP:0, TN:0, FP:0, FN:0}
    j = 0

    for i in folds:
        test = i
        train = [x for y in folds for x in y if y != test]
        test_tree(header, train, test, results)

    return results

def pass_fail_all_test(table, predict_col, header):
    newtab = []
    for student in table:
        newrow = []
        for i, h in enumerate(header):
            if i != predict_col and 'score' not in h:
                newrow.append(student[i])
        newtab.append(newrow + [student[predict_col]])
    return newtab

def c_matrix(tp, tn, fp, fn):  
      
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

def main():
    tname = 'graded_performance_v3.csv'
    students = u.read_table(tname)
    header = students[0]
    header_one = students[0][:-6]

    students = students[1:]
    k = 10

    math_table = pass_fail_all_test(students, 7, header)
    reading_table = pass_fail_all_test(students, 8, header)
    writing_table = pass_fail_all_test(students, 9, header)
    
    m = k_cross(header_one + ['pass_math'], math_table, k)
    r = k_cross(header_one + ['pass_reading'], reading_table, k)
    w = k_cross(header_one + ['pass_writing'], writing_table, k)

    print("\n\n-----------------------------------------------------------------")
    print("              Decision Tree Pass/Fail Predictions")
    print("-----------------------------------------------------------------")
    print("\n\n                       Math Scores")
    print("-----------------------------------------------------------------")
    c_matrix(m[TP], m[TN], m[FP], m[FN])
    print("-----------------------------------------------------------------")
    print("\n\n                      Reading Scores")
    print("-----------------------------------------------------------------")
    c_matrix(r[TP], r[TN], r[FP], r[FN])
    print("-----------------------------------------------------------------")
    print("\n\n                      Writing Scores")
    print("-----------------------------------------------------------------")
    c_matrix(w[TP], w[TN], w[FP], w[FN])
main()