import random
import numpy as np
import math

TP = 'true_pos'
TN = 'true_neg'
FP = 'false_pos'
FN = 'false_neg'

def read_students(fname):
    f = open(fname, 'r')
    lines = [x.strip('\n').strip(' ').split(',') for x in f]
    header = lines[0]
    insts = [[int(x) for x in y] for y in lines[1:]]
    return header, insts

def dot(x, y):
    return sum([x[i] * y[i] for i in range(len(x))])

def one_inst(weights, instance, r):
    actual = instance[-1]
    yt = dot(weights, instance[:-1])
    wnew = [w + (r * (actual - yt)) for w in weights]
    # print('    >', wnew)
    return [w + (r * (actual - yt)) for w in weights]

def rand_weights(n):
    return [random.randint(0, 100)/100 for x in range(n)]

def train_perceptrons(table, r):
    n = len(table[0])
    weights = rand_weights(n-1)
    for inst in table:
        weights = one_inst(weights, inst, r)
    return weights

def threshold_test(weights, table):
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
            # print('>>>>> newacc=', acc)
            bestacc = acc
    return bestt, bestweights

def k_cross_var(table, r, k, numtrials):
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

def main():
    f = 'perf_nums_v2.csv'
    r = 0.1
    k = 10
    numtrials = 10
    header, insts = read_students(f)
    print(k_cross_var(insts, r, k, numtrials))
main()