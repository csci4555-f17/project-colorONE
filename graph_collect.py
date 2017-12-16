import matplotlib.pyplot as plt
import numpy as np

accessNames = [
    "accesses_project_while.txt",
    "accesses_sean2_function5.txt",
    "accesses_sean2_list3.txt",
]

hybridNames = [
    "speed_collection.txt",
    "speed_collection2.txt",
]

for f in accessNames:
    i = 0
    avg = 0
    best = float('inf')
    worst = -1
    for line in open(f):
        v = int(line.split(',')[1])
        if i == 0:
            raw_ilp = v
        elif  i < 6:
            if v < best:
                best  = v
            if v > worst:
                worst = v
            avg += v
        elif i == 6:
            hybrid_25 = v
        elif i == 7:
            hybrid_50 = v
        elif i == 8:
            hybrid_75 = v
        i += 1

    avg = avg / 5.
    N = 7
    ind = np.arange(N)
    width = 0.35
    fig, ax = plt.subplots()
    data = [raw_ilp, best, worst, avg, hybrid_25, hybrid_50, hybrid_75]
    rects1 = ax.bar(ind, data, width, color='r')
    ax.set_ylabel("Memory Accesses")
    ax.set_title(f)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(('Base ILP', 'Best Greedy', 'Worst Greedy', 'Avg Greedy', 'Hybrid 25', 'Hybrid 50', 'Hybrid 75'))
    plt.ylim((0, hybrid_75*1.25))
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.3f' % float(height), ha='center', va='bottom')
    autolabel(rects1)
    plt.savefig("accesses_plot_{}.png".format(f), bbox_inches='tight')

for f in hybridNames:
    i = 0
    for line in open(f):
        v = float(line.split(',')[1])
        if i == 0:
            greedy = v
        elif i == 1:
            raw_ilp = v 
        elif i == 2:
            hybrid_25 = v
        elif i == 3:
            hybrid_50 = v
        elif i == 4:
            hybrid_75 = v
        i += 1

    N = 5
    ind = np.arange(N)
    width = 0.35
    fig, ax = plt.subplots()
    data = [greedy, raw_ilp, hybrid_25, hybrid_50, hybrid_75]
    rects1 = ax.bar(ind, data, width, color='r')
    ax.set_ylabel("Runtime (Hybrid)")
    ax.set_title(f)
    ax.set_xticks(ind + width/2)
    ax.set_xticklabels(('Greedy', 'Base ILP', 'Hybrid 25', 'Hybrid 50', 'Hybrid 75'))
    plt.ylim((0, raw_ilp*1.25))
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width()/2., 1.05*height, '%.3f' % float(height), ha='center', va='bottom')
    autolabel(rects1)
    plt.savefig("runtime_plot_{}.png".format(f), bbox_inches='tight')
