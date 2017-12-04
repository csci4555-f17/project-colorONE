def func1():
	while 5:
		print 4

class potato:
	y = 2

x = (potato if True else func1)()
print x.y
