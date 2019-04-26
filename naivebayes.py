import utils
# import math
# import tabulate
# from copy import deepcopy
# import numpy as np

def main():
    table = utils.read_table("StudentsPerformance.csv")
    header = table.pop(0)
    
    clean_data(table, header)

def clean_data(table, header):
    '''
    get rid of quotation marks on values, change strings to ints, add categorical classes
    '''
    strip_quotation_marks(table)
    scores_to_numeric(table, header)
    add_categorical_scores(table, header, "\"math score\"")
    add_categorical_scores(table, header, "\"reading score\"")
    add_categorical_scores(table, header, "\"writing score\"")

def strip_quotation_marks(table):
    '''
    since all values in raw table are stored in quotation marks,
    walk through values and strip them
    '''
    for row in table:
        for index,value in enumerate(row):
            new_value = value.strip("\"")
            row[index] = new_value

def scores_to_numeric(table, header):
    '''
    since scores are stored as strings but we want to use them as ints,
    convert scores in each row to numeric int values
    '''
    math_index = header.index("\"math score\"")
    reading_index = header.index("\"reading score\"")
    writing_index = header.index("\"writing score\"")

    for row in table:
        row[math_index] = int(row[math_index])
        row[reading_index] = int(row[reading_index])
        row[writing_index] = int(row[writing_index])

def add_categorical_scores(table, header, score_type):
    '''
    since we will be performing Naive Bayes classification, we need categorical classes
    appends rows to table with categorical values 1-10
    '''
    # use equal width bins to set categories
    score_index = header.index(score_type)
    scores = utils.get_column(table, score_index)
    cutoffs = utils.compute_equal_widths_cutoffs(scores, 10)

    # add new column to header
    header.append(score_type + " class")

    # for each row, find the right cutoff point and append int as class
    for row in table:
        cutoff_counter = 0
        while row[score_index] > cutoffs[cutoff_counter]:
            cutoff_counter += 1
        row.append(cutoff_counter)

def classify_using_naive_bayes(table, header, test_instance, class_label, attr_list):
    '''
    uses Naive Bayes to classify an unseen instance given a training set
    parameter table is the training data to use
    parameter header is a list of the attributes in table
    parameter test_instance is the test instance to be classified which does not have class_label attribute
    parameter class_label is the name (string) of the class
    parameter attr_list is a list of the attributes to use for classification
    returns the predicted class label for the test instance
    '''
    # get list of all possible class values
    classes = utils.get_unique_items(utils.get_column(table, header.index(class_label)))
    
    # initialize posteriors table with row for each class value, column for each attr
    posteriors_table = [[0 for value in classes] for value in attr_list]
    
    prior = 0 # will hold prior for given class value
    class_probabilities = [0 for value in classes] # should be parallel list to classes
    
    # for each class value (eg: on time, late, very late, cancelled), compute posteriors 
    #    then multiply to get P(class | test_instance)
    for class_value in classes:
        class_index = classes.index(class_value)
        
        # compute prior probability for current class value
        prior = prior_probability(table, header, class_label, class_value)

        # compute posteriors (intersect probability of each attribute in attr_list and current class)
        for attr_value in attr_list:
            attr_index = attr_list.index(attr_value)
            posteriors_table[attr_index][class_index] = (intersect_probability(table, header.index(class_label), class_value, \
            attr_index, attr_value))
        
        # compute conditional probability for current class
        class_probabilities[class_index] = compute_class_probability(prior, posteriors_table, class_index)

    # returns the class with the largest proportional probability given Naive Bayes classification
    max_class_index = class_probabilities.index(max(class_probabilities))
    return classes[max_class_index]

def prior_probability(table, header, class_label, class_value):
    '''
    computes the prior probability of a given class label
    '''
    count = 0
    class_index = header.index(class_label)
    for row in table:
        if row[class_index] == class_value:
            count += 1
    return count / len(table)

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
    for row in table:
        if row[class_index] == class_value:
            class_count += 1
            if row[attr_index] == attr_value:
                intersect_count += 1
    
    return (intersect_count / class_count)

def compute_class_probability(prior, posteriors_table, class_index):
    '''
    computes P(class | test_instance) by multiplying the prior and each posterior value
    '''
    class_probability = 1
    for posterior in posteriors_table:
        class_probability = class_probability * posterior[class_index]
    class_probability = class_probability * prior
   
    # if no instances causing 0 value, replace with prior
    if class_probability == 0:
        class_probability = prior
    
    return class_probability

if __name__ == "__main__":
    main()