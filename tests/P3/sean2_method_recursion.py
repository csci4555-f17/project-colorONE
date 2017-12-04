class M:
    def __init__(self, x):
        self.x = x
    def recurse(self):
        if self.x:
            print self.x
            self.x = self.x + -1
            self.recurse()
        else:
            print self.x

m = M(5)
m.recurse()
