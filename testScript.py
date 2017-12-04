import time
import os
import subprocess
import matplotlib
from compile import compilepy

input_file = "tests/sean2_simple_add.py"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), False, (True, True, True))
end = time.time()
print "Greedy compile time:", end-start, "seconds"

#ILP here

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (True, True, True))
end = time.time()
print "ILP compile time with all optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (False, True, True))
end = time.time()
print "ILP compile time with no de optimiztoin:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (True, False, True))
end = time.time()
print "ILP compile time with no static optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (True, True, False))
end = time.time()
print "ILP compile time with no mom optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (True, False, False))
end = time.time()
print "ILP compile time with no static, mem optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (False, True, False))
end = time.time()
print "ILP compile time with no de, mem optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (False, False, False))
end = time.time()
print "ILP compile time with no static, de, mem optimization:", end-start, "seconds"

start = time.time()
compilepy(input_file, input_file.replace('.py', '.s'), True, (False, False, True))
end = time.time()
print "ILP compile time with no de, static optimization:", end-start, "seconds"