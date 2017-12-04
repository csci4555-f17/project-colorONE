import time
import os
import subprocess
import matplotlib.pyplot as plt
from compile import compilepy
import numpy as np

input_files = ["tests/sean2_simple_add.py", "tests/P1/sean2_list3.py"]#, "tests/P1/sean2_list8.py", "tests/P1/sean2_list9.py", "tests/P2/sean2_function_test1.py", "tests/P2/sean2_function_test2.py", "tests/P2/sean2_function5.py"]
greedy = []
ilpAll = []
ilpDe = []
ilpStatic = []
ilpDeSta = []


for input_file in input_files:
	start = time.time()
	compilepy(input_file, input_file.replace('.py', '.s'), False, (True, True, True))
	end = time.time()
	greedy.append(end-start)
	print "Greedy compile time:", end-start, "seconds"

	#ILP here

	start = time.time()
	compilepy(input_file, input_file.replace('.py', '.s'), True, (True, True, True))
	end = time.time()
	ilpAll.append(end-start)
	print "ILP compile time with all optimization:", end-start, "seconds"

	start = time.time()
	compilepy(input_file, input_file.replace('.py', '.s'), True, (False, True, True))
	end = time.time()
	ilpDe.append(end-start)
	print "ILP compile time with no de optimiztoin:", end-start, "seconds"

	start = time.time()
	compilepy(input_file, input_file.replace('.py', '.s'), True, (True, False, True))
	end = time.time()
	ilpStatic.append(end-start)
	print "ILP compile time with no static optimization:", end-start, "seconds"


	start = time.time()
	compilepy(input_file, input_file.replace('.py', '.s'), True, (False, False, True))
	end = time.time()
	ilpDeSta.append(end-start)
	print "ILP compile time with no de, static optimization:", end-start, "seconds"


for i in range(len(greedy)):
	data = [greedy[i], ilpAll[i], ilpDe[i], ilpStatic[i], ilpDeSta[i]]
	N = 5
	ind = np.arange(N)
	width = 0.35
	fig, ax = plt.subplots()
	rects1 = ax.bar(ind, data, width, color='r')
	ax.set_ylabel('Times (seconds)')
	ax.set_title(input_files[i] + " performance")
	ax.set_xticks(ind + width / 2)
	ax.set_xticklabels(('Greedy', 'ILP All', 'ILP no De', 'ILP no Static', 'ILP None'))
	plt.ylim((0, ilpDe[i]*1.25))
	def autolabel(rects):
	    """
	    Attach a text label above each bar displaying its height
	    """
	    for rect in rects:
	        height = rect.get_height()
	        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
	                '%.3f' % float(height),
	                ha='center', va='bottom')
	autolabel(rects1)
	#plt.show()
	plt.savefig(str(i) + "perfGraph.png", bbox_inches='tight')


