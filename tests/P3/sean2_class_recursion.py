x = 3
class B:
    def __init__(self):
        print 2
    def caw(self, x):
        print x

class A(B):
    print x
    def __init__(self):
        self.a = 3
    def val(self, x):
        return 0 if x == 0 else 1+A.val(self, x+-1)

y = A()
x = y.val(3)
print x
