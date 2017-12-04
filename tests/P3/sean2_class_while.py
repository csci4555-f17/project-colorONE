class W:
    def __init__(self):
        self.iters = 10
    def func(self):
        x = self.iters
        while x:
            print x
            x = x + -1

w = W()
w.func()
