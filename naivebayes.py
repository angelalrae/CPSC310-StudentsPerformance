import utils
import tabulate

def main():
    table = utils.read_table("StudentsPerformance.csv")
    header = table.pop(0)
    
    clean_data(table, header)
    
    # split data
    test_set = utils.get_test_set(table, int(len(table) /3))
    train_set = utils.get_training_set(table, test_set)
    
    # make prediction about math score class
    class_label = "math score class"
    attributes = ['gender', 'race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']
    for test_instance in test_set:
        prediction = classify_using_naive_bayes(train_set, header, test_instance, class_label, attributes)
        print("prediction:", prediction, "actual:", test_instance[header.index("math score class")])

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

if __name__ == "__main__":
    main()