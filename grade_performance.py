
GRADES = {6:'D', 7:'C', 8:'B', 9: 'A'}

def clean_data(fname, ofname):
    fin = open(fname, 'r')
    fout = open(ofname, 'w')

    fout.write(fin.readline().replace(' ', '_')[:-1] + ",pass_math,pass_reading,pass_writing\n")
    for line in fin:
        line = [x.strip('"').strip('\n') for x in line.strip(' ').strip("'").split(',')]
        ofline = "%s,%d," % (line[0], edlevel(line[2])) + ','.join(line[3:-3])
        pfs = ''
        for x in line[-3:]:
            g = grade(x)
            pfs += (',' + pf(g))
            ofline += (',' + g)
        fout.write(ofline + pfs + '\n')

def edlevel(parent_ed):
    college = ["bachelor's degree", "master's degree"]
    college_2 = ["some college", "associate's degree"]
    hs = ["high school", "some high school"]
    if parent_ed in college:
        return 3
    if parent_ed in college_2:
        return 2
    else:
        return 1

def grade(score):
    g = int(score.strip('"')) // 10
    if g in GRADES:
        return GRADES[g]
    elif g < 6:
        return 'F'
    else:
        return 'A'

def pf(grade):
    if grade == 'F':
        return '0'
    return '1'




def main():
    fin = 'StudentsPerformance.csv'
    fout = 'graded_performance_a.csv'
    clean_data(fin, fout)

main()
