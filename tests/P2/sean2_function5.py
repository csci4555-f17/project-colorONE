x=3
y = lambda x:x+x
z = lambda x:x+x
def Return(x,y):
	x=1
	y=lambda x:x+x+x
	return y(x)+z(x)
print Return(1,2)
	
