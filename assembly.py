import os
from compiler.ast import *

class IndCall:
    def __str__(self):
        return "IndCall({}, {})".format(self.node, self.args)
    def __repr__(self):
        return "IndCall({}, {})".format(self.node, self.args)
    def __init__(self, node, args):
        self.node = node
        self.args = args

class Je:
    def __repr__(self):
        return "je {}".format(self.target)
    def __init__(self, target):
        self.target = target

class Jmp:
    def __repr__(self):
        return "jmp {}".format(self.target)
    def __init__(self, target):
        self.target = target

class IfA:
    def __repr__(self):
        thenStr = ""
        for i in self.then:
            thenStr += "\n\t{}".format(i)
        elseStr = ""
        for i in self.else_:
            elseStr += "\n\t{}".format(i)
        return "If(True; then{}\nelse{}".format(thenStr, elseStr)
        #return "If({}, {}, {}".format(self.test, self.then, self.else_)
    def __init__(self, test, then, else_):
        self.test = test
        self.then = then
        self.else_ = else_

class SetNe:
    def __str__(self):
        return "setne {}".format(self.reg)
    def __init__(self, reg):
        self.reg = reg

class SetE:
    def __str__(self):
        return "sete {}".format(self.reg)
    def __init__(self, reg):
        self.reg = reg

class Xorl:
    def __str__(self):
        return "xorl {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Andl:
    def __str__(self):
        return "andl {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Sarl:
    def __str__(self):
        return "sarl {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Orl:
    def __str__(self):
        return "orl {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Shl:
    def __str__(self):
        return "shll {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Cmpl:
    def __str__(self):
        return "cmpl {}, {}".format(self.source, self.dest)
    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Label:
    def __str__(self):
        return "{}:".format(self.name)
    def __init__(self, name):
        self.name = name

class FuncLabel:
    def __str__(self):
        return "${}".format(self.name)
    def __init__(self, name):
        if name == "t0":
            raise Exception("it happened boi")
        if not isinstance(name, str):
            raise Exception("ho boi")
        self.name = name

class Var:
    def __str__(self):
        return self.name
    
    def __init__(self, name, color=None, sticky=False):
        self.name = name
        self.color = color
        self.sticky = sticky

class Reg:
    def __str__(self):
        return "%{}".format(self.name)

    def __init__(self, name):
        self.name = name

class Mem:
    def __str__(self):
        return "{}(%ebp)".format(self.offset)

    def __init__(self, offset=0):
        if not isinstance(offset, int):
            raise Exception("it's in mem")
        self.offset = offset

class AConst:
    def __str__(self):
        return "${}".format(self.value)

    def __init__(self, value):
        if isinstance(value, str):
            raise Exception("bad number!")
        self.value = value


class Pushl:
    def __str__(self):
        return "pushl {}".format(self.arg)

    def __init__(self, arg):
        self.arg = arg

class Popl:
    def __str__(self):
        return "popl {}".format(self.arg)

    def __init__(self, arg):
        self.arg = arg

class Movl:
    def __str__(self):
        return "movl {}, {}".format(self.source, self.dest)

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Movzbl:
    def __str__(self):
        return "movzbl {}, {}".format(self.source, self.dest)

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Subl:
    def __str__(self):
        return "subl {}, {}".format(self.source, self.dest)

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Addl:
    def __str__(self):
        return "addl {}, {}".format(self.source, self.dest)

    def __init__(self, source, dest):
        self.source = source
        self.dest = dest

class Negl:
    def __str__(self):
        return "negl {}".format(self.arg)

    def __init__(self, arg):
        self.arg = arg

class Call:
    def __str__(self):
        return "call {}".format(self.target)

    def __init__(self, target):
        self.target = target

class FuncName:
    def __str__(self):
        return "FuncName({})".format(self.name)
    def __init__(self, name):
        self.name = name

class CallInd:
    def __str__(self):
        return "call *{}".format(self.target)
    def __init__(self, target):
        self.target = target

class Leave:
    def __str__(self):
        return "leave"

    def __init__(self):
        pass

class Ret:
    def __str__(self):
        return "ret"

    def __init__(self):
        pass

# extra AST nodes

class Let(Node):
    def __str__(self):
        return "Let({}, {}, {})".format(self.var, self.sub, self.expr)
    def __repr__(self):
        return "Let({}, {}, {})".format(self.var, self.sub, self.expr)
    def __init__(self, var, sub, expr):
        self.var = var
        self.sub = sub
        self.expr = expr

class GetTag(Node):
    def __repr__(self):
        return "GetTag({})".format(self.arg)
    def __init__(self, arg):
        self.arg = arg

class InjectFrom(Node):
    def __repr__(self):
        return "InjectFrom({}, {})".format(self.typ, self.arg)
    def __init__(self, typ, arg):
        self.typ = typ
        self.arg = arg

class ProjectTo(Node):
    def __repr__(self):
        return "ProjectTo({}, {})".format(self.typ, self.arg)
    def __init__(self, typ, arg):
        self.typ = typ
        self.arg = arg
