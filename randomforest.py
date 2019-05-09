# Allison Fellger and Angela Rae
# 5.9.2019

import math
import utils2 as utils
import utils as u
import numpy as np
import random
import tabulate

LEAF = 'leaf'
SPLIT = 'split'
UNSPLIT = 'unsplit'

class Forest(object):
    def __init__(self, m):
        self.m = m
        self.forest_list = []
        self.min_index = None

    def add_tree(self, tree, confusion_matrix):
        '''
        parameter tree is a TreeNode object
        parameter confusion_matrix is a tuple containing tp, tn, fp, and fn
        '''
        # store each new tree with it's confusion matrix as a 
        # dictionary in the forest list
        new_tree = {}
        new_tree["tree"] = tree
        new_tree["con_mat"] = confusion_matrix 
        # confusion matrix not currently used, but could be used
        #   to implement track record voting instead of simple majority
        new_tree["accuracy"] = get_current_accuracy(confusion_matrix)
        self.forest_list.append(new_tree)

    def min_accuracy(self):
        '''
        iterates through all trees and checks if accuracy is less than current min
        sets min_index value and returns minimum of accuracy values
        '''
        min_acc = self.forest_list[0]["accuracy"]
        new_min_index = 0
        for index,tree in enumerate(self.forest_list):
            accuracy = tree["accuracy"]
            if accuracy < min_acc:
                min_acc = accuracy
                new_min_index = index

        self.min_index = new_min_index
        return min_acc

    def replace(self, new_tree, confusion_matrix):
        '''
        remove old tree from forest, add new one, recalculate min accuracy
        '''
        self.forest_list.pop(self.min_index)
        self.add_tree(new_tree, confusion_matrix)
        self.min_accuracy()

    def classify(self, instance):
        '''
        classifies an instance with each tree, then uses simple majority voting
        to determine class to return
        '''
        predictions = []
        for tree in self.forest_list:
            predictions.append([tree["tree"].classify(instance)])

        return utils.majority_vote(predictions)


class TreeNode(object):
    def __init__(self, table, header, first=True, full_table=None):
        self.branches = {}
        self.table = table
        self.node_type = None
        self.split_index = None
        self.leaf_class = None
        self.header = header

        # to initialize, append all ?? to full table
        if first:
            full_table = []
            for att_index, _ in enumerate(header[:-1]):
                full_table.append(utils.unique([row[att_index] for row in table]))
            
        # get list of all class values
        classes = [x[-1] for x in table]
        c_types = utils.unique(classes)
        
        # if only one class, add leaf node
        if len(c_types) == 1:
            ut = utils.unique_table(self.table)
            self.node_type = LEAF
            self.leaf_class = c_types[0]

        # otherwise, use entropy to determine attribute index to split
        else:
            self.split_index = max_gain(table, header)
            # max_gain returns -1 if there is only one attr value in the current table

            if self.split_index != -1:
                # split on index with greatest information gain, then iterate over each attr value
                split_vals = utils.unique(table, col=self.split_index)
                self.node_type = SPLIT
                branch_tabs = [[y for y in table if y[self.split_index] == x] for x in split_vals]
                for i, branch in enumerate(branch_tabs):
                    self.branches[split_vals[i]] = TreeNode(branch, header, first=False, full_table=full_table)
            else:
                # if only one attribute value left, create leaf node
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


def max_gain(table, header):
    '''
        calculate all possible information gains and return the index of the
        attribute that will maximize info gain
    '''
    i_gains = {}
    for i, _ in enumerate(header[:-1]):
        # check if only one attribute value left
        #   if multiple get the information gain for each attribute
        if not utils.unanimous(table, i):
            i_gains[info_gain(table, i)] = i
    if len(i_gains) != 0:
        return i_gains[max(i_gains)]
    else:
        # if len == 0, unanimous class
        return -1


def info_gain(table, att_i):
    '''
        calculate information gain for one attribute
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

def test_tree(header, training_table, test_table):
    '''
        given training and test sets, generate a decision tree and
        return a tuple containing the count of true positives, true
        negatives, false positives and false negatives (confusion matrix)
    '''
    # generate decision tree
    model = TreeNode(training_table, header)
    confusion_matrix = [[0 for x in range(4)] for y in range(4)]

    # use test_table to generate confusion matrix to return
    for instance in test_table:
        predicted = model.classify(instance) - 1
        actual = instance[-1] - 1
        if predicted != -1:
            confusion_matrix[actual][predicted] += 1
        
    return model, confusion_matrix
        
def compute_bootstrap_sample(table):
    '''
    given a datatable, computes a bootstrap sample
    returns the sample (to be used as training set) 
      and the remainder (to be used as validation or test set)
    '''
    n = len(table)
    sample = []

    # loops through table and appends n random instances (with replacement) to list
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])

    # computes remainder or instances not chosen for sample
    remainder = []
    for row in table:
        if row not in sample:
            remainder.append(row)

    return sample, remainder

def stratify(table, header, class_label, k):
    '''
    parameters table and header are the data table and a list of the attributes in order, respectively
    parameter class_label is a string in the header representing the column to use to as class
    parameter k is the number of folds/partitions to make
    returns stratified_list a list of tables
    '''
    # group table by class label, partitioned table is a list of tables
    partitioned_table = u.group_by(table, header, class_label)
    
    # for each partition, loop through rows and append to different table in stratified list (also a list of tables)
    stratified_list = [[] for i in range(k)]
    for partition in partitioned_table:
        index = 0
        for row in partition:
            stratified_list[index].append(row)
            index = (index + 1) % k

    return stratified_list

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

def get_current_accuracy(con_mat):
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
    
    return acc(tp, tn, fp, fn)

def print_mat_accuracy(con_mat):
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
    
    print_c_matrix(tp, tn, fp, fn)

def one_mpg_acc(i, con_mat, total):
    tp = con_mat[i][i]
    fp = sum(con_mat[i]) - tp
    fn = sum([x[i] for x in con_mat]) - tp
    tn = total - (tp + fp + fn)
    return tp, tn, fp, fn

def print_c_matrix(tp, tn, fp, fn):  
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
    print('              Precision : %.5f\n' % precision(tp, tn, fp, fn))
    print('              Recall : %.5f\n' % recall(tp, tn, fp, fn))

def acc(tp, tn, fp, fn):
    '''
        calculate accuracy (correct classifications / all classifications)
    '''
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    return correct/ttl
    
def s_err(tp, tn, fp, fn):
    '''
        calculate standard error
    '''
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    err = math.sqrt((correct/ttl * incorrect/ttl) / ttl)
    return err

def precision(tp, tn, fp, fn):
    return tp / (tp + fp)

def recall(tp, tn, fp, fn):
    return tp / (tp + fn)

def compute_stratified(table, header, k):
    '''
    generates stratified test and remainder sets
    '''
    stratified_list = stratify(table, header, "AvgScore", k) #returns a list of stratified sets
    
    # k-1/k goes to remainder
    remainder = stratified_list[0]
    for i in (1, k-1):
        remainder.extend(stratified_list[i]) 

    # 1 section goes to test set
    test_set = stratified_list[-1] 

    return test_set, remainder

def main():
    # import table, get header, add average scores
    table_name = 'StudentsPerformance.csv'
    students = u.read_table(table_name)
    header = students[0][:-3] + ['AvgScore']
    students = group_scores(students[1:])
    
    # get stratified list for sample and test set for ensemble
    test_set, remainder = compute_stratified(students, header, 3)
        
    n = 100 # number of decision trees to generate
    m = 80 # number of best trees to save
    mForest = Forest(m) # initialize Forest object

    # create N decision trees using bootstrap sample to train
    # save best M in forest
    for i in range(n):
        # use bagging to get training set, then create tree
        train_set, validation_set = compute_bootstrap_sample(remainder)
        tree, temp_matrix = test_tree(header, train_set, validation_set)
            # temp matrix is a tuple containing tp, tn, fp, and fn

        # once m trees, check accuracy of new tree before adding it to forest
        if (i+1) > m:
            current_accuracy = get_current_accuracy(temp_matrix)
            if current_accuracy > mForest.min_accuracy():
                mForest.replace(tree, temp_matrix)
        else:
            mForest.add_tree(tree, temp_matrix)

    # test unseen instances using forest
    confusion_matrix = [[0 for x in range(4)] for y in range(4)]        
    for instance in test_set:
        predicted = mForest.classify(instance) - 1
        actual = instance[-1] - 1 #subtract 1 to match value to index

        if predicted != -1:
            confusion_matrix[actual][predicted] += 1
    print_mat_accuracy(confusion_matrix)

if __name__ == "__main__":
    main()