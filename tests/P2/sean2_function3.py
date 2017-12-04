f = lambda x: x+x
def ff1(x):
	return x(2) + x(2)
print ff1(f)
