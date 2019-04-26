import utils
import tabulate
import numpy as np 
from copy import deepcopy

def main():
    table = utils.read_table("StudentsPerformance.csv")
    header = table.pop(0)
    
    clean_data(table, header)
    
    # split data
    test_set = utils.get_test_set(table, int(len(table) /3))
    train_set = utils.get_training_set(table, test_set)
    
    # make prediction about math score class
    class_label = "writing score class"
    attributes = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
#     for test_instance in test_set:
#         prediction = classify_using_naive_bayes(train_set, header, test_instance, class_label, attributes)
        # print("prediction:", prediction, "actual:", test_instance[header.index("math score class")])

    subsampling_accuracy(table, header, 3, class_label, attributes)

def clean_data(table, header):
    '''
    get rid of quotation marks on values, change strings to ints, 
    add categorical classes
    '''
    strip_quotation_marks_table(table)
    strip_quotation_marks_list(header)
    scores_to_numeric(table, header)
    add_categorical_scores(table, header, "math score")
    add_categorical_scores(table, header, "reading score")
    add_categorical_scores(table, header, "writing score")

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
    parameter test_instance is an instance of the test set to be classified
    parameter class_label is the name (string) of the class
    parameter attr_list is a list of the attributes to use for classification
    returns the predicted class label for the test instance
    '''
    # get list of all possible class values
    classes = utils.get_unique_items(utils.get_column(table, header.index(class_label)))
    # initialize list of class probabilities
    class_probabilities = [0 for value in classes] # should be parallel list to classes
    
    # for each class, compute posteriors then multiply to get P(class | test_instance)
    for class_value in classes:
        current_class_index = classes.index(class_value)
        calculate_posterior(table, header, class_label, class_value, test_instance, \
            attr_list, current_class_index, class_probabilities)
        
    # returns the class with the largest proportional probability given Naive Bayes classification
    max_class_index = class_probabilities.index(max(class_probabilities))
    return classes[max_class_index]

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

def subsampling_accuracy(table, header, k, class_label, attr_list):
    '''
    tests the accuracy of Naive Bayes classifier using random subsampling to create test sets
    '''
    accuracies = []
    
    # For k trials, partition data into train and test then use linear regression and kNN 
    #    to predict for test set. Compute accuracy and error rate for both sets of predictions
    for i in range(k):
        # split table into train and test sets
        train, test = utils.compute_holdout_partitions(table)

        instance_accuracy = compute_naive_bayes_accuracy(train, test, header, attr_list, class_label)
        accuracies.append(instance_accuracy)
    
    # compute average accuracy
    accuracy = np.mean(accuracies)

    print('\n===========================================\nPredictive Accuracy\
    \n===========================================')
    print("Random Subsample (k=10, 2:1 Train/Test)")
    print("Naive Bayes: accuracy = ", round(accuracy, 2), ", error rate = ", round(1-accuracy, 2))

def compute_naive_bayes_accuracy(train, test, header, attributes_list, class_label):
    '''
    for a training/test set pair, computes the accuracy of the naive bayes classification
    parameter train is the training set of data (a table) for linear regression
    parameter test is the test set of data (a table) to classify
    parameters table and header are the data table and a list of the attributes in order, respectively
    returns the accuracy
    '''
    predictions = []

    # for each instance in the training set, compute naive bayes classifications
    for instance in test:
        predictions.append(classify_using_naive_bayes(train, header, instance, "math score class", attributes_list))

    # get actual values
    actuals = utils.get_column(test, header.index("math score class"))
    
    # compute accuracy using predicted values and actuals
    accuracy = compute_accuracy(predictions, actuals)

    return accuracy

def compute_accuracy(predictions, actuals):
    '''
    given parallel lists of predicted classes and actual classes, computes the accuracy
    returns the average of the class accuracies
    '''

    # get list of all possible classifications from predictions and actuals
    categories = utils.get_unique_items(deepcopy(actuals), deepcopy(predictions))
    categories = sorted(categories)
    
    # create confusion matrix (actual vs. predicted table)
    confusion_matrix = create_confusion_matrix(categories, actuals, predictions)
    
    # for each class label, compute the accuracy
    accuracies = []
    for class_label in categories:
        class_index = categories.index(class_label)
        accuracy = compute_class_accuracy(confusion_matrix, class_index)
        accuracies.append(accuracy)
    
    # return the average of the class accuracies
    return np.mean(accuracies)

def create_confusion_matrix(categories, actuals, predictions):
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

def compute_class_accuracy(table, class_index):
    '''
    uses the binary accuracy equation (accuracy = TP + TN / P + N) to compute the accuracy of a single class
    parameter table is the predictions versus actuals table
    parameter class_index is an int, the index of the class to compute accuracy for
    returns the class accuracy as a double
    '''
    true_positives = table[class_index][class_index]
    
    # compute true negatives (not classified as class and actually not in the class)
    true_negatives = 0
    for r_index,row in enumerate(table):
        for v_index,value in enumerate(row):
            if r_index != class_index and v_index != class_index:
                true_negatives += value

    # sum all values in table
    total = 0
    for row in table:
        for value in row:
            total += value
    
    return (true_positives + true_negatives) / total
if __name__ == "__main__":
    main()