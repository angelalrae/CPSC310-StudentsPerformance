import matplotlib.pyplot as plt
import numpy as np

def students(fname):
    f = open(fname, 'r')
    lines = [[y.strip('"') for y in x.strip('\n').strip(' ').split(',')] for x in f]
    
    header = lines[0]
    return header, lines[1:]

def getcol(data, col):
    return [x[col] for x in data]


def make_dot_chart(table, s_att, s_ops, s_labels, chart_title):
    '''
        Create a dot/strip chart of frequency based on att
    '''
    title = "Score Distribution by " + chart_title
    fname = chart_title.lower().replace(' ', '_') + '_plot.pdf'

    m1 = [int(x[-3].strip('"')) for x in table if x[s_att] == s_ops[0]]
    m2 = [int(x[-3].strip('"')) for x in table if x[s_att] == s_ops[1]]

    r1 = [int(x[-2].strip('"')) for x in table if x[s_att] == s_ops[0]]
    r2 = [int(x[-2].strip('"')) for x in table if x[s_att] == s_ops[1]]

    w1 = [int(x[-1].strip('"')) for x in table if x[s_att] == s_ops[0]]
    w2 = [int(x[-1].strip('"')) for x in table if x[s_att] == s_ops[1]]

    gps = [m1, m2, r1, r2, w1, w2]
    y_vals = [[y + 1 for i in range(len(group))] for y, group in enumerate(gps)]

    plt.figure()
    plt.title(title)
    plt.xlabel("Raw Exam Score out of 100")
    avxs = [np.mean(x) for x in gps]
    avys = [y_vals[x][0] for x in range(len(avxs))]
    colors = ['olive', 'purple', 'orange', 'crimson',  'slateblue', 'mediumturquoise']
    for i, g in enumerate(gps):
        plt.scatter(g, y_vals[i], marker='.', s=500, alpha=0.05, color=colors[i])

    plt.scatter(avxs,avys, marker='x', s=250, alpha=1.0, c='black')
    ytks = avys
    ylbs = ['Math-' + s_labels[0], 'Math-' + s_labels[1],
            'Reading-' + s_labels[0], 'Reading-' + s_labels[1],
            'Writing-' + s_labels[0], 'Writing-' + s_labels[1],]
    plt.yticks(ticks=ytks, labels=ylbs)
    
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

def main():
    f = 'StudentsPerformance.csv'
    h, s = students(f)
    make_dot_chart(s, 0, ['female', 'male'], ["Female", "Male"], "Gender")
    make_dot_chart(s, 3, ['standard', 'free/reduced'], ["Standard", "Free/Reduced"], "Lunch Status")
    make_dot_chart(s, 4, ['completed', 'none'], ["Prep Course", "No Prep Course"], "Preparation Course Completion")

main()