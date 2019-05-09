# Alli Fellger and Angela Rae
# filename: perceptron_net.py
#   reads from an input file 'perf_nums_v2.csv' and prints out three confusion matrices
#   representing the predictions made by a single-layer perceptron neural network as to
#   whether a student passed or failed the math, reading and writing exams documented
#   in 'perf_nums_v2.csv'

import random
import numpy as np
import math

TP = 'true_pos'
TN = 'true_neg'
FP = 'false_pos'
FN = 'false_neg'

def read_students(fname):
    '''
        Read in the student instances from the file titled fname
        return header, instances
    '''
    f = open(fname, 'r')
    lines = [x.strip('\n').strip(' ').split(',') for x in f]
    header = lines[0]
    insts = [[int(x) for x in y] for y in lines[1:]]
    return header, insts

def dot(x, y):
    '''
        return dot product of x and y
    '''
    return sum([x[i] * y[i] for i in range(len(x))])

def one_inst(weights, instance, r):
    '''
        return list of adjusted weights
    '''
    actual = instance[-1]
    yt = dot(weights, instance[:-1])
    return [w + (r * (actual - yt)) for w in weights]

def rand_weights(n):
    '''
        return n random weights
    '''
    return [random.randint(0, 100)/100 for x in range(n)]

def train_perceptrons(table, r):
    '''
        modify a set of random weights for every instance in the
        table, at learning rate r
    '''
    n = len(table[0])
    weights = rand_weights(n-1)
    for inst in table:
        weights = one_inst(weights, inst, r)
    return weights

def threshold_test(weights, table):
    '''
        Given a set of weights, return the optimal threshold
        for pass/fail.
    '''
    bestacc = 0
    besterr = 0
    bestt = 0
    for h in range(0, 100):
        t = h / 100
        acc, err = assess_err(weights, table, t)
        if acc > bestacc:
            bestacc = acc
            besterr = err
            bestt = t
    return bestacc, besterr, bestt

def learn(table, r, k):
    '''
        Find the best pass/fail threshold and best set
        of weights for table over k trials with learning
        rate r.
    '''
    bestacc = 0
    bestt = 0
    bestweights = []
    for _ in range(1, k + 1):
        random.shuffle(table)
        weights = train_perceptrons(table, r)
        acc, _, t = threshold_test(weights, table)
        if acc > bestacc:
            bestweights = weights
            bestt = t
            bestacc = acc
    return bestt, bestweights

def k_cross_var(table, r, k, numtrials):
    '''
        Use k-cross validation to test the model with 
        numtrials trials per training set, k folds and
        learning rate r.
    '''
    results = {TP:0, TN:0, FP:0, FN:0}

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

    for test in folds:
        train = [x for y in folds for x in y if y != test]
        t, weights = learn(train, r, numtrials)
        for inst in test:
            result = classify(weights, inst, t)
            if result == inst[-1]:
                if result == 1:
                    results[TP] += 1
                else:
                    results[TN] += 1
            else:
                if result == 1:
                    results[FP] += 1
                else:
                    results[FN] += 1
    c_matrix(results)
    return results, acc(results), s_err(results)

def c_matrix(r):  
    '''
        Print out the confusion matrix based on the dictionary r
    '''
    tp = r[TP]
    tn = r[TN]
    fp = r[FP]
    fn = r[FN]

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
    print('              Accuracy     : %.5f' % acc(r))
    print('              Standard Err : %.5f\n' % s_err(r))

def classify(weights, instance, t):
    '''
        classify the instance as 0 or 1 based on
        the weight set and the threshold t.
    '''
    pred = dot(weights, instance[:-1])
    if pred < t:
        return 0
    return 1

def acc(r):
    '''
        calculate accuracy (correct classifications / all classifications)
    '''
    tp = r[TP]
    tn = r[TN]
    fp = r[FP]
    fn = r[FN]
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    return correct/ttl

def s_err(r):
    '''
        calculate standard error
    '''
    tp = r[TP]
    tn = r[TN]
    fp = r[FP]
    fn = r[FN]
    correct = tp + tn
    incorrect = fp + fn
    ttl = correct + incorrect
    err = math.sqrt((correct/ttl * incorrect/ttl) / ttl)
    return err

def assess_err(weights, table, t):
    '''
        calculate the accurracy and standard error
        of classifying the predictions by using 
        weights and pass/fail threshold t.
    '''
    results = {TP:0, TN:0, FP:0, FN:0}
    for inst in table:
        result = classify(weights, inst, t)
        if result == inst[-1]:
            if result == 1:
                results[TP] += 1
            else:
                results[TN] += 1
        else:
            if result == 1:
                results[FP] += 1
            else:
                results[FN] += 1
    return acc(results), s_err(results)

def final_col_sort(instances, predcol):
    '''
        relocate the predcol to the end of each row within
        instances.
    '''
    newints = []
    for x in instances:
        newints.append(x[:predcol] + x[predcol+1:] + [x[predcol]])
    return newints

def main():
    f = 'perf_nums_v2.csv'
    r = 0.1
    k = 10
    numtrials = 10
    header, insts = read_students(f)
    mathtab = final_col_sort(insts, 4)
    readtab = final_col_sort(insts, 5)
    writetab = final_col_sort(insts, 6)

    print("\n\n-----------------------------------------------------------------")
    print("            Perceptron Net Pass/Fail Predictions")
    print("-----------------------------------------------------------------")
    print("\n\n                       Math Scores")
    print("-----------------------------------------------------------------")
    k_cross_var(mathtab, r, k, numtrials)
    print("-----------------------------------------------------------------")
    print("\n\n                      Reading Scores")
    print("-----------------------------------------------------------------")
    k_cross_var(readtab, r, k, numtrials)
    print("-----------------------------------------------------------------")
    print("\n\n                      Writing Scores")
    print("-----------------------------------------------------------------")
    k_cross_var(writetab, r, k, numtrials)
main()