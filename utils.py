##########################################################################
# Angela Rae and Alli Fellger
# Class: CPSC 310.01
##########################################################################

import numpy as np
import matplotlib.pyplot as plt 
import random
import math

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
        table.append(item_list);

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
    
    for i in range(instances):
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
    all code copied from Gina's notes
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