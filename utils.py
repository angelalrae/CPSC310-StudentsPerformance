##########################################################################
# Angela Rae and Alli Fellger
# Class: CPSC 310.01
##########################################################################

import numpy as np
import matplotlib.pyplot as plt 
import random
import math
from copy import deepcopy

def read_table(filename):
    '''
    reads file and converts it to a 2D list
    parameter filename is the file containing a comma separated dataset to be read
    returns 2D list containing rows from file
    '''
    
    table = []
    item_list = []
    infile = open(filename, "r")
    
    # read each line in infile and append to table as row (ie 1d list)
    lines = infile.readlines()

    for line in lines: 
        # get rid of newline char
        line = line.strip() # strips whitespace characters
        #break line into individual strings using comma delimiter
        values = line.split(",")
        # print(values)
        convert_to_numeric(values)
        item_list = values
        table.append(item_list)

    infile.close()
    return table

def convert_to_numeric(values):
    '''
    walks through each value in values list and tries to convert each to an int/float
    parameter values is a list created from a line in data file
    '''
    for i in range(len(values)):
        try:
            numeric_val = float(values[i])
            # if this runs, success
            values[i] = numeric_val
        except ValueError:
            pass

def get_column(table, index):
    '''
    gets column of data from table
    parameter table is the data table to read from
    parameter index is the index of the column to be returned
    returns list of values
    '''
    column = []

    for row in table:
        if row[index] == 'NA':
            pass
        else:
            column.append(row[index])
        
    return column

def get_test_set(table, instances):
    '''
    generates a test set of data
    parameter table is the data table (2D list) to choose instances from
    parameter instances is the number of instances to return
    returns a sample 2D list of instances from table
    '''
    test_set = []
    
    for _ in range(instances):
        index = random.randint(0, len(table)-1)
        test_set.append(table[index])
    
    return test_set

def get_training_set(table, test_set):
    '''
    '''
    train = []
    for instance in table:
        if instance not in test_set:
            train.append(instance)

    return train


def compute_linear_regression(x, y):
    '''
    computes m and b values for the linear regression of x and y
    parameters x and y are lists of equal length, where y is a responding variable to x
    returns m and b
    '''
    mean1 = np.mean(x)
    mean2 = np.mean(y)
    m = sum([(x[i] - mean1) * (y[i] - mean2) for i in range(len(x))]) / \
        sum([(x[i] - mean1) ** 2 for i in range(len(x))]) 
    b = mean2 - m * mean1
    return m,b

def normalize_list(list):
    '''
    normalizes values in list to range 0-1
    '''
    new_list = []
    
    for item in list:
        x = (int(item) - min(list)) / (max(list) - min(list))
        new_list.append(x)

    return new_list

def print_row(row):
    '''
    prints instance/row with comma between values
    '''
    print_string = "instance: "
    for index,item in enumerate(row):
        print_string += str(item)
        if index != (len(row)-1):
            print_string += ", "
    
    print(print_string)

def get_cat_frequencies(column):
    '''
    for categorical dataset, gets categories and count per category
    parameter column is a list of data
    returns list of categories and list of frequencies per category (parallel lists)
    '''
    categories = []
    frequencies = []
    
    for value in column:
        if value in categories:
            frequencies[categories.index(value)] += 1
        else:
            categories.append(value)
            frequencies.append(1)

    return categories, frequencies

def compute_holdout_partitions(table):
    '''
    Copied from U4 C Classifier Evaluation notes
    '''
    # randomize the table
    randomized = table[:] # copy the table
    n = len(table)
    for i in range(n):
        # pick an index to swap
        j = random.randrange(0, n) # random int in [0,n) 
        randomized[i], randomized[j] = randomized[j], randomized[i]
    # return train and test sets
    split_index = int(2 / 3 * n) # 2/3 of randomized table is train, 1/3 is test
    return randomized[0:split_index], randomized[split_index:]

def get_unique_items(list1, list2 = None):
    '''
    returns list with all unique values in list1 and list2 (i.e. no duplicates)
    '''
    if list2 is None:
        return list(set(list1))
    
    else:
        for value in list2:
            list1.append(value)
        return list(set(list1))

def group_by(table, header, grouping_attr):
    '''
    '''
    # get list of all classes to partition on and initialize partitioned_table to 2D list of length of # of classes
    group_by_index = header.index(grouping_attr)
    grouping_column = get_column(table, group_by_index)
    classes, class_freq = get_cat_frequencies(grouping_column)
    partitioned_table = [[] for i in range(len(classes))] #classes will be header for partitioned_table
    
    for row in table:
        for class_index,class_label in enumerate(classes):
            if row[group_by_index] == class_label:
                partitioned_table[class_index].append(row)
    
    return partitioned_table

def gaussian(x, mean, sdev):
    '''
    parameter x is attribute value from continuous attribute
    parameter mean is the mean of values in the column
    parameter sdev is the standard deviation of values in the column
    returns the posterior/conditional probability
    function copied from Gina's notes
    '''
    first, second = 0, 0
    if sdev > 0:
        first = 1 / (math.sqrt(2 * math.pi) * sdev)
        second = math.e ** (-((x - mean) ** 2) / (2 * (sdev ** 2)))
    return first * second

def compute_equal_widths_cutoffs(values, num_bins):
    '''
    converts continuous dataset to categorical by creating equal width bins
    parameter values is a list of data
    parameter num_bins is the number of bins to sort values into
    returns list of cutoff values
    '''
    width = (max(values) - min(values)) / num_bins # returns float

    # create list of upper bounds
    cutoffs = list(np.arange(min(values) + width, max(values) + width, width))
    # round each decimal to one place before returning
    cutoffs = [round(cutoff, 1) for cutoff in cutoffs]
    
    return cutoffs

def strip_quotation_marks_table(table):
    '''
    since all values in raw table are stored in quotation marks,
    walk through values and strip them
    '''
    for row in table:
        strip_quotation_marks_list(row)
        
def strip_quotation_marks_list(list):
    '''
    walk through list and strip quotation marks
    '''
    for index,value in enumerate(list):
        new_value = value.strip("\"")
        list[index] = new_value

def scores_to_numeric(table, header):
    '''
    since scores are stored as strings but we want to use them as ints,
    convert scores in each row to numeric int values
    '''
    math_index = header.index("math score")
    reading_index = header.index("reading score")
    writing_index = header.index("writing score")

    for row in table:
        row[math_index] = int(row[math_index])
        row[reading_index] = int(row[reading_index])
        row[writing_index] = int(row[writing_index])

def add_grades(table, header, score_type):
    # add new column to header
    score_index = header.index(score_type)
    header.append(score_type + " class")

     # for each row, add grade from
    for row in table:
        grade = get_grade(row[score_index])
        row.append(grade)

def get_grade(score):
    if score >= 90:
        return "A"
    elif score >= 80:
        return "B"
    elif score >= 70:
        return "C"
    elif score >= 60:
        return "D"
    else:
        return "F"

def calculate_posterior(table, header, class_label, class_value, test_instance, \
    attr_list, current_class_index, class_probabilities):
    '''
    calculates the posterior probability P(class | instance) using
    '''
    # compute prior probability for current class value
    prior = prior_probability(table, header, class_label, class_value)

    # compute conditional probabilities 
    # (i.e. intersect probability of each test instance value for attrs in attr_list and current class)
    posteriors_for_class = [0 for attr in attr_list]
    for attribute in attr_list:
        attr_index = header.index(attribute)
        attr_value = test_instance[attr_index]
        posteriors_for_class[attr_index] = (intersect_probability(table, header.index(class_label), \
            class_value, attr_index, attr_value))
    
    # compute conditional probability for current class
    class_probabilities[current_class_index] = compute_class_probability(prior, posteriors_for_class, current_class_index)

def prior_probability(table, header, class_label, class_value):
    '''
    computes the prior probability of a given class value and dataset
    '''
    class_count = 0
    class_index = header.index(class_label)
    for row in table:
        if row[class_index] == class_value:
            class_count += 1

    return class_count / len(table)

def intersect_probability(table, class_index, class_value, attr_index, attr_value):
    '''
    computes the probability of both values in the table = P(X n A)/P(X)
    parameter table is the dataset
    parameters class_value and attr_value are the values to check table rows against
    parameters class_index and attr_index are int representing the position of the values in the table/header
    returns the probability (float) of both values
    '''
    class_count = 0
    intersect_count = 0

    # loop through table to count instances where class and attr value match test instance
    for row in table:
        if row[class_index] == class_value:
            class_count += 1
            if row[attr_index] == attr_value:
                intersect_count += 1
    
    return (intersect_count / class_count)

def compute_class_probability(prior, posteriors_list, class_index):
    '''
    computes P(class | test_instance) by multiplying the prior and each posterior value
    '''
    class_probability = prior
    for posterior in posteriors_list:
        class_probability = class_probability * posterior
        
    # if no instances causing 0 value, replace with prior
    if class_probability == 0:
        class_probability = prior
    
    return class_probability


def create_confusion_matrix(predictions, actuals):
    '''
    given parallel lists of predicted classes and actual classes, creates a confusion matrix
    '''

    # get list of all possible classifications from predictions and actuals
    categories = get_unique_items(deepcopy(actuals), deepcopy(predictions))
    categories = sorted(categories)
    
    # create confusion matrix (actual vs. predicted table)
    raw_matrix = create_raw_matrix(categories, actuals, predictions)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    # create confusion matrix with pass/fail numbers
    for a_index,actuals_row in enumerate(raw_matrix):
        for p_index in range(len(actuals_row)):
            if a_index < 4 and p_index < 4:
                tp += raw_matrix[a_index][p_index]
            elif a_index < 4:
                fn += raw_matrix[a_index][p_index]
            elif p_index < 4:
                fp += raw_matrix[a_index][p_index]
            else:
                tn += raw_matrix[a_index][p_index]

    return (tp, tn, fp, fn)

def create_raw_matrix(categories, actuals, predictions):
    '''
    creates a confusion maxtrix of actual versus predicted values
    parameter categories is a list of all unique classes
    parameter actuals is a list of the actual class values from the dataset
    parameter predictions is a list of the predicted values from the classifier
    returns a 2D list of actuals x predictions
    '''
    # initialize table with zeros for each cell
    confusion_matrix = [[0 for i in range(len(categories))] for i in range(len(categories))]

    # loop through list indices, 
    #   find the actual category and predicted category, then increment appropriate cell
    for index in range(len(actuals)):
        actual_index = 'A' # will throw error if somehow incorrect
        predicted_index = 'A'
        for class_label in categories:
            if class_label == actuals[index]:
                actual_index = categories.index(class_label)
            if class_label == predictions[index]:
                predicted_index = categories.index(class_label)
        
        confusion_matrix[actual_index][predicted_index] += 1
        
    return confusion_matrix

def print_confusion_matrix(tp, tn, fp, fn):  
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
