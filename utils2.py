
def unique(tab_list, col=None):
    '''
        Return a list of the unique items in tab_list 
    '''
    if col == None:
        return list(set(tab_list))
    else:
        u = [x[col] for x in tab_list]
        return list(set(u))

def unanimous(table, col):
    '''
        Check if table[col] contains only one unique value.
    '''
    c_results = [x[col] for x in table]
    types = len(unique(c_results))
    return types == 1

def majority_vote(table):
    '''
        return the most frequent classification in the table.
    '''
    classes = [x[-1] for x in table]
    types = unique(classes)
    counts = {x:sum([1 for c in classes if c == x]) for x in types}
    ma = d_max(counts)
    mi = d_min(counts)

    return d_max(counts)

def unique_table(table):
    '''
        Return the unique values of each row in a table.
    '''
    un = []
    for i, col in enumerate(table[0]):
        un.append(unique([x[i] for x in table]))
    return un

def match_table(t1, t2):
    '''
        Check if two tables are equivalent.
    '''
    for item in t1:
        if item not in t2:
            return False
    for item in t2:
        if item not in t1:
            return False
    return True

def d_max(n_dic):
    '''
        Find the maximum value in a dictionary and return the corresponding key.
    '''
    maxkey = None
    maxval = None
    for k in n_dic:
        if maxval == None or n_dic[k] > maxval:
            maxkey = k
            maxval = n_dic[k]
    return maxkey

def d_min(n_dic):
    '''
        Find the minimum value in a dictionary and return the corresponding key.
    '''
    minkey = None
    minval = None
    for k in n_dic:
        if minval == None or n_dic[k] < minval:
            minkey = k
            minval = n_dic[k]
    return minkey

def add_tables(t1, t2):
    '''
        Add two tables together (must be same dimensions)
    '''
    outt = [[0 for x in t1[0]] for x in t1]
    for i, row in enumerate(t1):
        for j, col in enumerate(row):
            outt[i][j] += (t2[i][j] + t1[i][j])
    return outt
