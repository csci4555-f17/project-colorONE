class A:
    def __init__(self, mark):
        self.mark = mark
    def val(self):
        return self.mark
    def decr(self):
        self.mark = self.mark + -1

t = A(5)
while t.val():
    print t.val()
    t.decr()
