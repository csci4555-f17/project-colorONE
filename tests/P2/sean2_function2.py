y = 4
def func1(x):
	y = 3
	return x+x+x+x+x+x+x+y
def func2(x):
	return func1(x)+func1(x)+func1(x)
print func2(y)+func2(3)
