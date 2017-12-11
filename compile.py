import argparse
import os
import compiler
from assembly import *
import random

from ilp_coloring import color_ilp, spillCost
from py_experiments import calculateAccesses

from compiler.ast import *

regMap = {
    1: "ecx",
    2: "edx",
    3: "eax",
    4: "ebx",
    5: "edi",
    6: "esi",
}

reservedFunctions = set([
    "__init__",
    "is_true",
    "exit",
    "equal",
    "not_equal",
    "add",
    "create_list",
    "set_subscript",
    "create_dict",
    "get_subscript",
    "input",
    "create_closure",
    "get_free_vars",
    "get_fun_ptr",
    "create_class",
    "set_attr",
    "get_attr",
    "has_attr",
    "get_function",
    "is_class",
    "create_object",
    "is_bound_method",
    "is_unbound_method",
    "get_receiver",
])

def prune_if(assemblyAst, ifIndex):
    newAst = []
    for i in range(len(assemblyAst)):
        instr = assemblyAst[i]

        if isinstance(instr, IfA):
            elseLabel = Label("else_label_{}".format(ifIndex))
            elseJmp = Je("else_label_{}".format(ifIndex))
            endLabel = Label("end_label_{}".format(ifIndex))
            endJmp = Jmp("end_label_{}".format(ifIndex))
            ifIndex += 1
            prunedElse, ifIndex = prune_if(instr.else_, ifIndex)
            prunedThen, ifIndex = prune_if(instr.then, ifIndex)
    
            newAst.append(elseJmp)
            for ti in prunedThen:
                newAst.append(ti)
            newAst.append(endJmp)
            newAst.append(elseLabel)
            for ei in prunedElse:
                newAst.append(ei)
            newAst.append(endLabel)

        elif isinstance(instr, While):
            loopLabel = Label("while_label_{}".format(ifIndex))
            endJmp = Je("end_label_{}".format(ifIndex))
            loopJmp = Jmp("while_label_{}".format(ifIndex))
            endLabel = Label("end_label_{}".format(ifIndex))
            ifIndex += 1

            newAst.append(loopLabel)
            prunedTest, ifIndex = prune_if(instr.test[0], ifIndex)
            for ins in prunedTest:
                newAst.append(ins)
            newAst.append(endJmp)

            prunedBody, ifIndex = prune_if(instr.body, ifIndex)

            for ti in prunedBody:
                newAst.append(ti)
            newAst.append(loopJmp)
            newAst.append(endLabel)

        else:
            newAst.append(instr)

    return newAst, ifIndex

def closure(n, index):
    if isinstance(n, Module):
        new_mod, flist, index = closure(n.node, index)
        return Module("", new_mod), flist, index

    elif isinstance(n, Stmt):
        new_stmt = []
        new_flist = []
        for i in n.nodes:
            temp, flist, index = closure(i, index)
            new_stmt.append(temp)
            new_flist += flist
        return Stmt(new_stmt), new_flist, index

    elif isinstance(n, str):
        return n, [], index 
    
    elif isinstance(n, Let):
        new_s, lflist, index = closure(n.sub, index)
        new_e, rflist, index = closure(n.expr, index)
        return Let(n.var, new_s, new_e), lflist+rflist, index

    elif isinstance(n, InjectFrom):
        new_in, flist, index = closure(n.arg, index)
        return InjectFrom(n.typ, new_in), flist, index

    elif isinstance(n, ProjectTo):
        new_pt, flist, index = closure(n.arg, index)
        return ProjectTo(n.typ, new_pt), flist, index

    elif isinstance(n, GetTag):
        new_gt, flist, index = closure(n.arg, index)
        return GetTag(new_gt), flist, index

    elif isinstance(n, Const):
        return n, [], index

    elif isinstance(n, Name):
        return n, [], index

    elif isinstance(n, AssName):
        return n, [], index

    elif isinstance(n, Add):
        new_l, flist_1, index = closure(n.left, index)
        new_r, flist_2, index = closure(n.right, index)
        return Add((new_l, new_r)), flist_1 + flist_2, index

    elif isinstance(n, And):
        new1, flist1, index = closure(n.nodes[0], index)
        new2, flist2, index = closure(n.nodes[1], index)
        return And([new1, new2]), flist1 + flist2, index
    
    elif isinstance(n, Or):
        new1, flist1, index = closure(n.nodes[0], index)
        new2, flist2, index = closure(n.nodes[1], index)
        return Or([new1, new2]), flist1 + flist2, index

    elif isinstance(n, UnarySub):
        new_U, flist, index = closure(n.expr, index)
        return UnarySub(new_U), flist, index

    elif isinstance(n, Not):
        new_n, flist, index = closure(n.expr, index)
        return Not(new_n), flist, index

    elif isinstance(n, Discard):
        new_d, flist, index = closure(n.expr, index)
        return Discard(new_d), flist, index

    elif isinstance(n, Printnl):
        new_p, flist, index = closure(n.nodes[0], index)
        return Printnl([new_p], None), flist, index 

    elif isinstance(n, Assign):
        rhs, rflist, index = closure(n.expr, index)
        lhs, lflist, index  = closure(n.nodes[0], index)
        return Assign([lhs], rhs), rflist + lflist, index
        
    elif isinstance(n, IfExp):
        te, tflist, index = closure(n.test, index)
        th, thflist, index = closure(n.then, index)
        el, elflist, index = closure(n.else_, index)
        return IfExp(te, th, el), tflist+thflist + elflist, index

    elif isinstance(n, If):
        te, tflist, index = closure(n.tests[0][0], index)
        th, thflist, index = closure(n.tests[0][1], index)
        el, elflist, index = closure(n.else_, index)
        return If([(te, th)], el), tflist+thflist + elflist, index

    elif isinstance(n, While):
        te, tflist, index = closure(n.test, index)
        tb, tblist, index = closure(n.body, index)
        return While(te, tb, n.else_), tflist+tblist, index

    elif isinstance(n, Compare):
        left, lflist, index = closure(n.expr, index)
        right, rflist, index = closure(n.ops[0][1], index)
        return Compare(left, [(n.ops[0][0], right)]), lflist+rflist, index


    elif isinstance(n, Subscript):
        ex, exflist, index = closure(n.expr, index)
        su, suflist, index = closure(n.subs[0], index)
        return Subscript(ex, 'OP_APPLY', [su]), exflist + suflist, index

    elif isinstance(n, CallFunc):
        if isinstance(n.node, Name):
            if n.node.name not in reservedFunctions:
                fun_ptr_call = CallFunc(Name("get_fun_ptr"), [Name("1234f")])
                free_vars_call = CallFunc(Name("get_free_vars"), [Name("1234f")])
                new_args = []
                new_funcs = []
                for e in n.args:
                    ne, nf, index = closure(e, index)
                    new_args.append(ne)
                    new_funcs += nf

                cnode = IndCall(fun_ptr_call, [free_vars_call] + new_args)
                lnode = Let(Name("1234f"), n.node, cnode)
                return lnode, new_funcs, index
            else:
                new_args = []
                new_funcs = []
                for e in n.args:
                    ne, nf, index = closure(e, index)
                    new_args.append(ne)
                    new_funcs += nf
                cnode = CallFunc(n.node, new_args)
                return cnode, new_funcs, index

        fun_ptr_call = CallFunc(Name("get_fun_ptr"), [Name("1234f")])
        free_vars_call = CallFunc(Name("get_free_vars"), [Name("1234f")])
        new_args = []
        new_funcs = []
        for e in n.args:
            ne, nf, index = closure(e, index)
            new_args.append(ne)
            new_funcs+=nf

        cnode = IndCall(fun_ptr_call, [free_vars_call] + new_args)
        closed_mid, mf, index = closure(n.node, index)
        lnode = Let(Name("1234f"), closed_mid, cnode)
        return lnode, new_funcs+mf, index

    elif isinstance(n, Lambda):
        globalname = "lambda_{}".format(index)
        index += 1
        fvs = free_vars(n) - reservedFunctions

        fv_args = [Name(s) for s in fvs]
        listCreate = InjectFrom(3, CallFunc(Name("create_list"), [InjectFrom(0, Const(len(fv_args)))]))
        if not len(fv_args):
            creationLet = listCreate
        else:
            cexpr = Name("1234list")
            for i in range(len(fv_args)-1, -1, -1):
                cexpr = Let(
                    Name("1234set"),
                    CallFunc(Name("set_subscript"), [Name("1234list"), InjectFrom(0, Const(i)), fv_args[i]]),
                    cexpr
                )
            creationLet = Let(Name("1234list"), listCreate, cexpr)

        global_arg = FuncName(globalname)

        args = [global_arg, creationLet]
        new_node = InjectFrom(3, CallFunc(Name("create_closure"), args))
    
        new_args = ["fvs"] + n.argnames
        fv_inits = []
        fv_assigns = [AssName(s, None) for s in fvs]
        for i,f in enumerate(fv_assigns):
            fv_inits.append(Assign([f], Subscript(Name("fvs"), "OP_APPLY", [InjectFrom(0, Const(i))])))
        new_body, subflist, index = closure(n.code, index)
        new_code = Stmt(fv_inits+new_body.nodes)
        new_func = Function(None, globalname, new_args, [], 0, "", new_code)

        return new_node,(subflist+[new_func]),index

    elif isinstance(n, Return): 
        re, flist, index = closure(n.value, index)
        return Return(re), flist, index

    else:
        raise Exception("unrecognized ast node in closure {}".format(n))

def heapify(n, freeVars):
    if isinstance(n, Module):
        statements = heapify(n.node, freeVars)
        return Module("", statements)

    elif isinstance(n, Stmt):
        new_statements = []
        for e in n.nodes:
            n_n = heapify(e, freeVars) 
            new_statements.append(n_n)

        return Stmt(new_statements)
    
    elif isinstance(n, Let):
        n_sub = heapify(n.sub, freeVars)
        n_expr = heapify(n.expr, freeVars)

        return Let(n.var, n_sub, n_expr)

    elif isinstance(n, InjectFrom):
        return InjectFrom(n.typ, heapify(n.arg, freeVars))

    elif isinstance(n, ProjectTo):
        return ProjectTo(n.typ, heapify(n.arg, freeVars))

    elif isinstance(n, GetTag):
        return GetTag(heapify(n.arg, freeVars))

    elif isinstance(n, Const):
        return n

    elif isinstance(n, Name):
        if n.name != "True" and n.name != "False":
            if n.name in freeVars:
                return Subscript(n, 'OP_APPLY', [InjectFrom(0, Const(0))])
            return n
        return n

    elif isinstance(n, Add):
        return Add((heapify(n.left, freeVars), heapify(n.right, freeVars)))

    elif isinstance(n, And):
        nodes = [heapify(n.nodes[0], freeVars), heapify(n.nodes[1], freeVars)]
        return And(nodes)
    
    elif isinstance(n, Or):
        nodes = [heapify(n.nodes[0], freeVars), heapify(n.nodes[1], freeVars)]
        return Or(nodes)

    elif isinstance(n, str):
        return n

    elif isinstance(n, UnarySub):
        return UnarySub(heapify(n.expr, freeVars))

    elif isinstance(n, Not):
        return Not(heapify(n.expr, freeVars))

    elif isinstance(n, Discard):
        return Discard(heapify(n.expr, freeVars))

    elif isinstance(n, Printnl):
        return Printnl([heapify(n.nodes[0], freeVars)], None)

    elif isinstance(n, Assign):
        rhs = heapify(n.expr, freeVars)
        lhs = n.nodes[0]
        if isinstance(lhs, AssName):
            if lhs.name in freeVars:
                lhs = Subscript(Name(lhs.name), 'OP_APPLY', [InjectFrom(0, Const(0))])
        else:
            lhs = heapify(lhs, freeVars)
        return Assign([lhs], rhs)

    elif isinstance(n, IfExp):
        te = heapify(n.test, freeVars)
        t = heapify(n.then, freeVars)
        e = heapify(n.else_, freeVars)
        return IfExp(te, t, e)
    
    elif isinstance(n, If):
        te = heapify(n.tests[0][0], freeVars)
        t = heapify(n.tests[0][1], freeVars)
        e = heapify(n.else_, freeVars)
        return If([(te, t)], e)

    elif isinstance(n, While):
        te = heapify(n.test, freeVars)
        t = heapify(n.body, freeVars)
        return While(te, t, n.else_)

    elif isinstance(n, Compare):
        l = heapify(n.expr, freeVars)
        r = heapify(n.ops[0][1], freeVars)

        return Compare(l, [(n.ops[0][0], r)])

    elif isinstance(n, Subscript):
        e = heapify(n.expr, freeVars)
        s = heapify(n.subs[0], freeVars)
        return Subscript(e, 'OP_APPLY', [s])

    elif isinstance(n, CallFunc):
        new_args = []
        for e in n.args:
            ee = heapify(e, freeVars)
            new_args.append(ee)
        return CallFunc(heapify(n.node, freeVars), new_args)

    elif isinstance(n, Lambda):
        body = heapify(n.code, freeVars)
        L = gather_assignments(n.code)
        P = n.argnames
        Pp = []
        Ph = []
        for s in P:
            if s in freeVars:
                Pp.append("{}_param".format(s))
                Ph.append(s)
            else:
                Pp.append(s)

        paramAllocs = []
        for i in range(len(Ph)):
            listCreate = InjectFrom(3, CallFunc(Name("create_list"), [InjectFrom(0, Const(1))]))
            cexpr = Let(
                Name("1234set"),
                CallFunc(Name("set_subscript"), [Name("1234list"), InjectFrom(0, Const(0)), InjectFrom(0, Const(0))]),
                Name("1234list")
            )
            creationLet = Let(Name("1234list"), listCreate, cexpr)
            paramAllocs.append(Assign([AssName(Ph[i],None)], creationLet))

        paramInits = []
        for i in range(len(Ph)):
            paramInits.append(Assign([Subscript(Name(Ph[i]), 'OP_APPLY', [Const(0)])], Name(Pp[i])))

        Lh = [s for s in L if s in freeVars]
        localInits = []
        for i in range(len(Lh)):
            listCreate = InjectFrom(3, CallFunc(Name("create_list"), [InjectFrom(0, Const(1))]))
            cexpr = Let(
                Name("1234set"),
                CallFunc(Name("set_subscript"), [Name("1234list"), InjectFrom(0, Const(0)), InjectFrom(0, Const(0))]),
                Name("1234list")
            )
            creationLet = Let(Name("1234list"), listCreate, cexpr)
            paramAllocs.append(Assign([AssName(Lh[i],None)], creationLet))

        new_code = Stmt(paramAllocs + paramInits + localInits + body.nodes)

        return Lambda(Pp, n.defaults, n.flags, new_code)

    elif isinstance(n, Return): 
        return Return(heapify(n.value, freeVars))

    else:
        raise Exception("unrecognized ast node in heapify {}".format(n))

def compute_free_vars(n, freeVars):
    if isinstance(n, Module):
        compute_free_vars(n.node, freeVars)

    elif isinstance(n, str):
        return

    elif isinstance(n, Stmt):
        fvars = []
        for e in n.nodes:
            compute_free_vars(e, freeVars) 
    
    elif isinstance(n, Let):
        compute_free_vars(n.sub, freeVars)
        compute_free_vars(n.expr, freeVars)

    elif isinstance(n, InjectFrom):
        compute_free_vars(n.arg, freeVars)

    elif isinstance(n, ProjectTo):
        compute_free_vars(n.arg, freeVars)

    elif isinstance(n, GetTag):
        compute_free_vars(n.arg, freeVars)

    elif isinstance(n, Const):
        return

    elif isinstance(n, Name):
        return

    elif isinstance(n, AssName):
        return

    elif isinstance(n, Add):
        compute_free_vars(n.left, freeVars)
        compute_free_vars(n.right, freeVars)

    elif isinstance(n, And):
        compute_free_vars(n.nodes[0], freeVars)
        compute_free_vars(n.nodes[1], freeVars)
    
    elif isinstance(n, Or):
        compute_free_vars(n.nodes[0], freeVars)
        compute_free_vars(n.nodes[1], freeVars)

    elif isinstance(n, UnarySub):
        compute_free_vars(n.expr, freeVars)

    elif isinstance(n, Not):
        compute_free_vars(n.expr, freeVars)

    elif isinstance(n, Discard):
        compute_free_vars(n.expr, freeVars)

    elif isinstance(n, Printnl):
        compute_free_vars(n.nodes[0], freeVars)

    elif isinstance(n, Assign):
        compute_free_vars(n.expr, freeVars)
        compute_free_vars(n.nodes[0], freeVars)

    elif isinstance(n, IfExp):
        compute_free_vars(n.test, freeVars)
        compute_free_vars(n.then, freeVars)
        compute_free_vars(n.else_, freeVars)

    elif isinstance(n, If):
        compute_free_vars(n.tests[0][0], freeVars)
        compute_free_vars(n.tests[0][1], freeVars)
        compute_free_vars(n.else_, freeVars)

    elif isinstance(n, While):
        compute_free_vars(n.test, freeVars)
        compute_free_vars(n.body, freeVars)

    elif isinstance(n, Compare):
        compute_free_vars(n.expr, freeVars)
        compute_free_vars(n.ops[0][1], freeVars)

    elif isinstance(n, Subscript):
        compute_free_vars(n.expr, freeVars)
        compute_free_vars(n.subs[0], freeVars)

    elif isinstance(n, CallFunc):
        for e in n.args:
            compute_free_vars(e, freeVars)
        compute_free_vars(n.node, freeVars)

    elif isinstance(n, Lambda):
        fvars = free_vars(n)
        fvars = fvars - reservedFunctions
        for v in fvars:
            freeVars.append(v)
        compute_free_vars(n.code, freeVars)

    elif isinstance(n, Return): 
        compute_free_vars(n.value, freeVars)

    else:
        raise Exception("unrecognized ast node in compute_free_vars {}".format(n))

def free_vars(n):
    if isinstance(n, Stmt):
        local_bindings = gather_assignments(n)        
        fvars = [free_vars(e) for e in n.nodes]
        fvars = reduce(lambda a,b: a|b, fvars, set([]))

        free = fvars - local_bindings
        return free

    if isinstance(n, str):
        return set([])

    elif isinstance(n, Const):
        return set([])

    elif isinstance(n, Name):
        if n.name == "True" or n.name == "False":
            return set([])
        else:
            return set([n.name])

    elif isinstance(n, Let):
        return (free_vars(n.expr) - set([n.var.name])) | free_vars(n.sub)

    elif isinstance(n, InjectFrom):
        return free_vars(n.arg)

    elif isinstance(n, ProjectTo):
        return free_vars(n.arg)

    elif isinstance(n, GetTag):
        return free_vars(n.arg)

    elif isinstance(n, Add):
        return free_vars(n.left) | free_vars(n.right)

    elif isinstance(n, And):
        return free_vars(n.nodes[0]) | free_vars(n.nodes[1])
    
    elif isinstance(n, Or):
        return free_vars(n.nodes[0]) | free_vars(n.nodes[1])

    elif isinstance(n, UnarySub):
        return free_vars(n.expr)

    elif isinstance(n, Not):
        return free_vars(n.expr)

    elif isinstance(n, Discard):
        return free_vars(n.expr)

    elif isinstance(n, Printnl):
        return free_vars(n.nodes[0])

    elif isinstance(n, Assign):
        if isinstance(n.nodes[0], AssName):
            return free_vars(n.expr)
        return free_vars(n.expr) | free_vars(n.nodes[0])

    elif isinstance(n, IfExp):
        return free_vars(n.test) | free_vars(n.then) | free_vars(n.else_)

    elif isinstance(n, If):
        return free_vars(n.tests[0][0]) | free_vars(n.tests[0][1]) | free_vars(n.else_)

    elif isinstance(n, While):
        return free_vars(n.test) | free_vars(n.body)

    elif isinstance(n, Compare):
        return free_vars(n.expr) | free_vars(n.ops[0][1])

    elif isinstance(n, Subscript):
        return free_vars(n.expr) | free_vars(n.subs[0])

    elif isinstance(n, CallFunc):
        f_args = [free_vars(e) for e in n.args]
        free_args = reduce(lambda a,b: a|b, f_args, set([]))
        return free_vars(n.node) | free_args

    elif isinstance(n, Lambda):
        free = free_vars(n.code) - set(n.argnames)
        return free

    elif isinstance(n, Return): 
        return free_vars(n.value)

    raise Exception("unrecognized ast node in free_vars {}".format(n))

def adjustAst(assemblyAst, failedVar, stackMap):
    newAst = []
    for i in range(len(assemblyAst)):
        instr = assemblyAst[i]

        if isinstance(instr, Pushl):
            if isinstance(instr.arg, Var):
                if instr.arg.name == failedVar:
                    memNode = Mem(stackMap[instr.arg.name])
                    newAst.append(Pushl(memNode))
                else:
                    newAst.append(instr)
            else:
                newAst.append(instr)

        elif isinstance(instr, IfA):
            elseAst = adjustAst(instr.else_, failedVar, stackMap)
            thenAst = adjustAst(instr.then, failedVar, stackMap)
            newAst.append(IfA(instr.test, thenAst, elseAst))

        elif isinstance(instr, While):
            testAst = adjustAst(instr.test[0], failedVar, stackMap)
            bodyAst = adjustAst(instr.body, failedVar, stackMap)
            newAst.append(While((testAst, instr.test[1]), bodyAst, None))

        elif isinstance(instr, Movl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                mnode = Movl(vnode, rnode)
                newAst.append(mnode)
            else:
                newAst.append(Movl(lnode, rnode))

        elif isinstance(instr, Movzbl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movzbl(lnode, vnode)
                newAst.append(mnode)
                mnode = Movzbl(vnode, rnode)
                newAst.append(mnode)
            else:
                newAst.append(Movzbl(lnode, rnode))

        elif isinstance(instr, Cmpl):
            lnode = instr.source
            rnode = instr.dest
            # assume Const isn't on rnode
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])

            if isinstance(lnode, AConst):
                if isinstance(rnode, Mem):
                    vnode = Var(name="spillVar", color=None, sticky=True)
                    mnode = Movl(rnode, vnode)
                    newAst.append(mnode)
                    newAst.append(Cmpl(lnode, vnode))
                else:
                    newAst.append(Cmpl(lnode, rnode))

            elif isinstance(lnode, Mem):
                if isinstance(rnode, Mem):
                    vnode = Var(name="spillVar", color=None, sticky=True)
                    mnode = Movl(rnode, vnode)
                    newAst.append(mnode)
                    newAst.append(Cmpl(lnode, vnode))
                else:
                    newAst.append(Cmpl(lnode, rnode))

            else:
                newAst.append(Cmpl(lnode, rnode))

        elif isinstance(instr, Addl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Addl(vnode, rnode))
            else:
                newAst.append(Addl(lnode, rnode))

        elif isinstance(instr, Andl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Andl(vnode, rnode))
            else:
                newAst.append(Andl(lnode, rnode))

        elif isinstance(instr, Orl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Orl(vnode, rnode))
            else:
                newAst.append(Orl(lnode, rnode))

        elif isinstance(instr, Xorl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Xorl(vnode, rnode))
            else:
                newAst.append(Xorl(lnode, rnode))

        elif isinstance(instr, Shl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Shl(vnode, rnode))
            else:
                newAst.append(Shl(lnode, rnode))

        elif isinstance(instr, Sarl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                if instr.source.name == failedVar:
                    lnode = Mem(stackMap[instr.source.name])
            if isinstance(instr.dest, Var):
                if instr.dest.name == failedVar:
                    rnode = Mem(stackMap[instr.dest.name])
            if isinstance(lnode, Mem) and isinstance(rnode, Mem):
                vnode = Var(name="spillVar", color=None, sticky=True)
                mnode = Movl(lnode, vnode)
                newAst.append(mnode)
                newAst.append(Sarl(vnode, rnode))
            else:
                newAst.append(Sarl(lnode, rnode))

        elif isinstance(instr, Negl):
            if isinstance(instr.arg, Var):
                if instr.arg.name == failedVar:
                    memNode = Mem(stackMap[instr.arg.name])
                    newAst.append(Negl(memNode))
                else:
                    newAst.append(instr)
            else:
                newAst.append(instr)

        elif isinstance(instr, SetE):
            newAst.append(instr)
        elif isinstance(instr, SetNe):
            newAst.append(instr)
        elif isinstance(instr, Call):
            #if instr.target == "t0":
            #    raise Exception("in adjust")
            newAst.append(instr)
        elif isinstance(instr, CallInd):
            if isinstance(instr.target, Var):
                if instr.target.name == failedVar:
                    memNode = Mem(stackMap[instr.target.name])
                    newAst.append(CallInd(memNode))
                else:
                    newAst.append(instr)
            else:
                newAst.append(instr)
        elif isinstance(instr, Label):
            newAst.append(instr)
        elif isinstance(instr, Popl):
            newAst.append(instr)
        elif isinstance(instr, Leave):
            newAst.append(instr)
        elif isinstance(instr, Ret):
            newAst.append(instr)
        else:
            raise Exception("{}".format(instr))
            newAst.append(instr)

    return newAst

def assignRegisters(assemblyAst, coloring):
    for i in range(len(assemblyAst)):
        instr = assemblyAst[i]

        if isinstance(instr, Pushl):
            if isinstance(instr.arg, Var):
                rnode = Reg(regMap[coloring[instr.arg.name][0]])
                assemblyAst[i] = Pushl(rnode)

        elif isinstance(instr, Label):
            pass

        elif isinstance(instr, Popl):
            pass

        elif isinstance(instr, Leave):
            pass

        elif isinstance(instr, Ret):
            pass

        elif isinstance(instr, IfA):
            assignRegisters(instr.then, coloring)
            assignRegisters(instr.else_, coloring)

        elif isinstance(instr, While):
            assignRegisters(instr.test[0], coloring)
            assignRegisters(instr.body, coloring)

        elif isinstance(instr, Movl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Movl(lnode, rnode)

        elif isinstance(instr, Movzbl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Movzbl(lnode, rnode)

        elif isinstance(instr, Cmpl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Cmpl(lnode, rnode)

        elif isinstance(instr, Addl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Addl(lnode, rnode)

        elif isinstance(instr, Negl):
            if isinstance(instr.arg, Var):
                rnode = Reg(regMap[coloring[instr.arg.name][0]])
                assemblyAst[i] = Negl(rnode)

        elif isinstance(instr, Shl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Shl(lnode, rnode)

        elif isinstance(instr, Sarl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Sarl(lnode, rnode)

        elif isinstance(instr, Andl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Andl(lnode, rnode)

        elif isinstance(instr, Orl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Orl(lnode, rnode)
        
        elif isinstance(instr, Xorl):
            lnode = instr.source
            rnode = instr.dest
            if isinstance(instr.source, Var):
                lnode = Reg(regMap[coloring[instr.source.name][0]])
            if isinstance(instr.dest, Var):
                rnode = Reg(regMap[coloring[instr.dest.name][0]])
            assemblyAst[i] = Xorl(lnode, rnode)

        elif isinstance(instr, SetE):
            continue

        elif isinstance(instr, SetNe):
            continue

        elif isinstance(instr, Call):
            if not isinstance(instr.target, str):
                raise Exception("wat")
            continue

        elif isinstance(instr, CallInd):
            tnode = instr.target
            if isinstance(instr.target, Var):
                tnode = Reg(regMap[coloring[instr.target.name][0]])
                assemblyAst[i] = CallInd(tnode)
                continue
            
            assemblyAst[i] = CallInd(tnode)

        else:
            raise Exception("{}".format(instr))

def color_fast(iGraph, colorMapping):
    colorSet = set([k for k in range(1, 7)])
    W = [k for k in colorMapping if colorMapping[k][0] is None]
    q = []
    adjMap = {}
    for i,w in enumerate(W):
        adjColors = set([colorMapping[k][0] for k in iGraph[w] if colorMapping[k][0] is not None])
        p = [len(adjColors), adjColors]
        adjMap[w] = p
        q.append((w, p))

    while q:
        q.sort(key=lambda x: x[1][0], reverse=True)
        top = q[0]
        ties = [top]
        for i in range(1, len(q)):
            if q[i][1][0] == top[1][0]:
                ties.append(q[i])
            else:
                break
        selected = random.choice(ties)
        acceptableColors = colorSet.difference(selected[1][1])

        if not len(acceptableColors):
            return False, selected[0], colorMapping

        e = colorMapping[selected[0]]
        col = set([list(acceptableColors)[0]])
        colorMapping[selected[0]] = (list(col)[0], e[1])

        for a in iGraph[selected[0]]:
            if '%' not in a:
                newSet = adjMap[a][1] | col
                adjMap[a][0] = len(newSet)
                adjMap[a][1] = newSet

        q.remove(selected)

    return True, None, colorMapping

def color(iGraph, colorMapping):
    W = [k for k in colorMapping if colorMapping[k][0] is None]
    while len(W):
        saturationList = []
        for w in W:
            adjColors = set([colorMapping[k][0] for k in iGraph[w] if colorMapping[k][0] is not None])
            saturation = len(adjColors)
            saturationList.append((saturation, w, adjColors))
        saturationList.sort(key=lambda x: x[0], reverse=True)
        
        highestSat = -1
        highest = []
        for sat,w,adjCol in saturationList:
            if highestSat == -1:
                highestSat = sat
                highest.append((w, adjCol))
            elif sat == highestSat:
                highest.append((w, adjCol))
            else:
                break

        selectedW = random.choice(highest)
        acceptableColors = set([k for k in range(1, 7)]).difference(selectedW[1])
        
        if not len(acceptableColors):
            return False,selectedW[0],colorMapping

        e = colorMapping[selectedW[0]]
        colorMapping[selectedW[0]] = (list(acceptableColors)[0], e[1])
        W.remove(selectedW[0])

    return True,None,colorMapping

def zombie(assemblyAst, startSet, iGraph, colorMapping):
    liveSet = []
    currentSet = set(startSet)

    for i in range(len(assemblyAst) - 1, -1, -1):
        instr = assemblyAst[i]
        if isinstance(instr, Pushl):
            if isinstance(instr.arg, Var):
                if instr.arg.name not in colorMapping:
                    colorMapping[instr.arg.name] = (instr.arg.color, instr.arg.sticky)
                currentSet = currentSet.union(set([instr.arg.name]))
        elif isinstance(instr, Movl) or isinstance(instr, Movzbl):
            if isinstance(instr.dest, Var):
                if instr.dest.name not in iGraph:
                    iGraph[instr.dest.name] = set([])
                if isinstance(instr.source, Var):
                    iGraph[instr.dest.name] = iGraph[instr.dest.name].union(set([k for k in currentSet if k != instr.dest.name and k != instr.source.name]))
                else:
                    iGraph[instr.dest.name] = iGraph[instr.dest.name].union(set([k for k in currentSet if k != instr.dest.name]))

                for k in currentSet:
                    if k != instr.dest.name:
                        if isinstance(instr.source, Var) and k == instr.source.name:
                            continue

                        if k not in iGraph:
                            iGraph[k] = set([])
                        iGraph[k] = iGraph[k].union(set([instr.dest.name]))

            if isinstance(instr.dest, Var):
                if instr.dest.name not in colorMapping:
                    colorMapping[instr.dest.name] = (instr.dest.color, instr.dest.sticky)
                currentSet = currentSet.difference(set([instr.dest.name]))
            if isinstance(instr.source, Var):
                if instr.source.name not in colorMapping:
                    colorMapping[instr.source.name] = (instr.source.color, instr.source.sticky)
                currentSet = currentSet.union(set([instr.source.name]))

        elif isinstance(instr, IfA):
            #Else
            elseAst = instr.else_
            elseSet = set(currentSet)
            elseSet = zombie(elseAst, elseSet, iGraph, colorMapping)

            #Then
            thenAst = instr.then
            thenSet = set(currentSet)
            thenSet = zombie(thenAst, thenSet, iGraph, colorMapping)

            currentSet = elseSet.union(thenSet)
            
            if isinstance(instr.test, Var):
                if instr.test.name not in colorMapping:
                    colorMapping[instr.test.name] = (instr.test.color, instr.test.sticky)
                currentSet = currentSet.union(set([instr.test.name]))


        elif isinstance(instr, While):
            bodyAst = instr.body
            bodySet = set(currentSet)

            tmpGraph = dict(iGraph)
            tmpColor = dict(colorMapping)

            if isinstance(instr.test[1], Var):
                if instr.test[1].name not in colorMapping:
                    colorMapping[instr.test[1].name] = (instr.test[1].color, instr.test[1].sticky)
                bodySet = bodySet.union(set([instr.test[1].name]))
            
            runSet = zombie(bodyAst, bodySet, tmpGraph, tmpColor)
            if isinstance(instr.test[1], Var):
                runSet = runSet.union(set([instr.test[1].name]))

            while (runSet - bodySet):
                tmpGraph = dict(iGraph)
                tmpColor = dict(colorMapping)
                bodySet = runSet
                runSet = zombie(bodyAst, bodySet, tmpGraph, tmpColor)
                if isinstance(instr.test[1], Var):
                    runSet = runSet.union(set([instr.test[1].name]))

           
            for k in tmpGraph:
                iGraph[k] = tmpGraph[k]
            for k in tmpColor:
                colorMapping[k] = tmpColor[k]

            currentSet = currentSet.union(runSet)

            currentSet = zombie(instr.test[0], currentSet, iGraph, colorMapping)
        
        elif isinstance(instr, Addl) or isinstance(instr, Andl) or isinstance(instr, Orl) or isinstance(instr, Xorl) or isinstance(instr, Shl) or isinstance(instr, Sarl):
            if isinstance(instr.dest, Var):
                if instr.dest.name not in iGraph:
                    iGraph[instr.dest.name] = set([])
                iGraph[instr.dest.name] = iGraph[instr.dest.name].union(set([k for k in currentSet if k != instr.dest.name]))

                for k in currentSet:
                    if k != instr.dest.name:
                        if k not in iGraph:
                            iGraph[k] = set([])
                        iGraph[k] = iGraph[k].union(set([instr.dest.name]))

            if isinstance(instr.dest, Var):
                if instr.dest.name not in colorMapping:
                    colorMapping[instr.dest.name] = (instr.dest.color, instr.dest.sticky)
                currentSet = currentSet.union(set([instr.dest.name]))
            if isinstance(instr.source, Var):
                if instr.source.name not in colorMapping:
                    colorMapping[instr.source.name] = (instr.source.color, instr.source.sticky)
                currentSet = currentSet.union(set([instr.source.name]))

        elif isinstance(instr, Cmpl):
            if isinstance(instr.dest, Var):
                if instr.dest.name not in colorMapping:
                    colorMapping[instr.dest.name] = (instr.dest.color, instr.dest.sticky)
                currentSet = currentSet.union(set([instr.dest.name]))
            if isinstance(instr.source, Var):
                if instr.source.name not in colorMapping:
                    colorMapping[instr.source.name] = (instr.source.color, instr.source.sticky)
                currentSet = currentSet.union(set([instr.source.name]))

        elif isinstance(instr, Negl):
            if isinstance(instr.arg, Var):
                if instr.arg.name not in iGraph:
                    iGraph[instr.arg.name] = set([])
                iGraph[instr.arg.name] = iGraph[instr.arg.name].union(set([k for k in currentSet if k != instr.arg.name]))

                for k in currentSet:
                    if k != instr.arg.name:
                        if k not in iGraph:
                            iGraph[k] = set([])
                        iGraph[k] = iGraph[k].union(set([instr.arg.name]))

            if isinstance(instr.arg, Var):
                if instr.arg.name not in colorMapping:
                    colorMapping[instr.arg.name] = (instr.arg.color, instr.arg.sticky)
                currentSet = currentSet.union(set([instr.arg.name]))

        elif isinstance(instr, Ret):
            pass

        elif isinstance(instr, Leave):
            pass
        elif isinstance(instr, Label):
            pass
        # yay tech debt!
        elif isinstance(instr, Popl):
            pass

        elif isinstance(instr, SetE):
            if "%eax" not in colorMapping:
                colorMapping["%eax"] = (1, True)
            iGraph["%eax"] = iGraph["%eax"].union(set([k for k in currentSet if k != "%eax"]))           
            for k in currentSet:
                if k != "%eax":
                    if k not in iGraph:
                        iGraph[k] = set([])
                    iGraph[k] = iGraph[k].union(set(["%eax"]))

        elif isinstance(instr, SetNe):
            if "%eax" not in colorMapping:
                colorMapping["%eax"] = (1, True)
            iGraph["%eax"] = iGraph["%eax"].union(set([k for k in currentSet if k != "%eax"]))           
            for k in currentSet:
                if k != "%eax":
                    if k not in iGraph:
                        iGraph[k] = set([])
                    iGraph[k] = iGraph[k].union(set(["%eax"]))
 
        elif isinstance(instr, Call):
            if "%ecx" not in colorMapping:
                colorMapping["%ecx"] = (1, True)
            if "%edx" not in colorMapping:
                colorMapping["%edx"] = (2, True)
            if "%eax" not in colorMapping:
                colorMapping["%eax"] = (3, True)

            if "%ecx" not in iGraph:
                iGraph["%ecx"] = set([])
            if "%edx" not in iGraph:
                iGraph["%edx"] = set([])
            if "%eax" not in iGraph:
                iGraph["%eax"] = set([])

            iGraph["%ecx"] = iGraph["%ecx"].union(set([k for k in currentSet if k != "%ecx"]))
            iGraph["%edx"] = iGraph["%edx"].union(set([k for k in currentSet if k != "%edx"]))
            iGraph["%eax"] = iGraph["%eax"].union(set([k for k in currentSet if k != "%eax"]))

            for k in currentSet:
                if k != "%ecx" and k != "%edx" and k != "%eax":
                    if k not in iGraph:
                        iGraph[k] = set([])
                    iGraph[k] = iGraph[k].union(set(["%ecx", "%eax", "%edx"]))


        elif isinstance(instr, CallInd):
            if "%ecx" not in colorMapping:
                colorMapping["%ecx"] = (1, True)
            if "%edx" not in colorMapping:
                colorMapping["%edx"] = (2, True)
            if "%eax" not in colorMapping:
                colorMapping["%eax"] = (3, True)

            if "%ecx" not in iGraph:
                iGraph["%ecx"] = set([])
            if "%edx" not in iGraph:
                iGraph["%edx"] = set([])
            if "%eax" not in iGraph:
                iGraph["%eax"] = set([])

            iGraph["%ecx"] = iGraph["%ecx"].union(set([k for k in currentSet if k != "%ecx"]))
            iGraph["%edx"] = iGraph["%edx"].union(set([k for k in currentSet if k != "%edx"]))
            iGraph["%eax"] = iGraph["%eax"].union(set([k for k in currentSet if k != "%eax"]))

            for k in currentSet:
                if k != "%ecx" and k != "%edx" and k != "%eax":
                    if k not in iGraph:
                        iGraph[k] = set([])
                    iGraph[k] = iGraph[k].union(set(["%ecx", "%eax", "%edx"]))

            if isinstance(instr.target, Var):
                if instr.target.name not in colorMapping:
                    colorMapping[instr.target.name] = (instr.target.color, instr.target.sticky)
                currentSet = currentSet.union(set([instr.target.name]))

        else:
            raise Exception(instr)
    return currentSet

def translate_ir(node, assemblyAst, strings):
    if isinstance(node, Module):
        translate_ir(node.node, assemblyAst, strings)

    elif isinstance(node, Function):
        paramMap = {}
        off = 8
        for a in node.argnames:
            paramMap[a] = off
            off += 4
        
        assemblyAst.append(Label(node.name))
        assemblyAst.append(Pushl(Reg("ebp")))
        assemblyAst.append(Movl(Reg("esp"), Reg("ebp")))
        for cs in ["ebx", "edi", "esi"]:
            assemblyAst.append(Pushl(Reg(cs)))

        bodyAst = []
        translate_ir(node.code, bodyAst, strings)

        for a in node.argnames:
            bodyAst = adjustAst(bodyAst, a, paramMap)

        for i in bodyAst:
            assemblyAst.append(i)

        if not isinstance(assemblyAst[-1], Ret):
            for cs in ["esi", "edi", "ebx"]:
                assemblyAst.append(Popl(Reg(cs)))
            assemblyAst.append(Leave())
            assemblyAst.append(Ret())

    elif isinstance(node, Return):
        pnode = None
        if isinstance(node.value, Const):
            pnode = AConst(node.value.value)
        elif isinstance(node.value, Name):
            pnode = Var(node.value.name)
        elif isinstance(node.value, FuncName):
            pnode = FuncLabel(node.value.name)

        assemblyAst.append(Movl(pnode, Reg("eax")))
        for cs in ["esi", "edi", "ebx"]:
            assemblyAst.append(Popl(Reg(cs)))
        assemblyAst.append(Leave())
        assemblyAst.append(Ret())
        
    elif isinstance(node, Stmt):
        for n in node.nodes:
            translate_ir(n, assemblyAst, strings)

    elif isinstance(node, CallFunc):
        addBack = 0
        for i in range(len(node.args)-1, -1, -1):
            arg = node.args[i]
            pnode = None
            if isinstance(arg, Const):
                pnode = AConst(arg.value)
            elif isinstance(arg, Name):
                pnode = Var(arg.name)
            elif isinstance(arg, str):
                pnode = "${}".format(strings[arg])
            else:
                raise Exception("bad node: {}".format(arg))
            assemblyAst.append(Pushl(pnode))
            addBack += 4
        callNode = Call(node.node.name)
        assemblyAst.append(callNode)
        if addBack:
            addNode = Addl(AConst(addBack), Reg("esp"))
            assemblyAst.append(addNode)

    elif isinstance(node, IndCall):
        addBack = 0
        for i in range(len(node.args)-1, -1, -1):
            arg = node.args[i]
            pnode = None
            if isinstance(arg, Const):
                pnode = AConst(arg.value)
            elif isinstance(arg, Name):
                pnode = Var(arg.name)
            elif isinstance(arg, FuncName):
                pnode = FuncLabel(arg.name)
            elif isinstance(arg, str):
                pnode = "${}".format(strings[arg])
            else:
                raise Exception("bad node type: {}".format(arg))
            assemblyAst.append(Pushl(pnode))
            addBack += 4
        if isinstance(node.node, Name):
            callNode = CallInd(Var(node.node.name))
        else:
            raise Exception("bad node: {}".format(node))
        assemblyAst.append(callNode)
        if addBack:
            addNode = Addl(AConst(addBack), Reg("esp"))
            assemblyAst.append(addNode)

    elif isinstance(node, IfA):
        # assembly for test
        testNode = None
        placNode = None
        if isinstance(node.test, Name):
            placNode = Var(node.test.name)
            testNode = Cmpl(AConst(0), placNode)
        elif isinstance(node.test, Const):
            placNode = Var("testTemp")
            movNode = Movl(AConst(node.test.value), placNode)
            assemblyAst.append(movNode)
            testNode = Cmpl(AConst(0), placNode)
        else:
            raise Exception("{}".format(node.test))

        # assembly for then
        thenAst = []
        translate_ir(node.then, thenAst, strings)

        # assembly for else 
        elseAst = []
        translate_ir(node.else_, elseAst, strings)

        assemblyAst.append(testNode)
   
        ifNode = IfA(placNode, thenAst, elseAst)

        assemblyAst.append(ifNode)

    elif isinstance(node, Printnl):
        if len(node.nodes) != 1:
            raise Exception('more than one print argument in flattened code!')

        if isinstance(node.nodes[0], Const):
            cnode = AConst(node.nodes[0].value)
            node = Pushl(cnode)
            assemblyAst.append(node)

        elif isinstance(node.nodes[0], Name):
            vnode = Var(node.nodes[0].name)
            pushNode = Pushl(vnode)
            assemblyAst.append(pushNode)

        else:
            raise Exception('print expression is not flat!')

        callNode = Call("print_any")
        assemblyAst.append(callNode)
        addNode = Addl(AConst(4), Reg("esp"))
        assemblyAst.append(addNode)
        
    elif isinstance(node, Assign):
        if len(node.nodes) != 1:
            raise Exception("more than one assignment to assign in flattened code!")
        
        if isinstance(node.expr, Const):
            cnode = AConst(node.expr.value)
            vnode = Var(node.nodes[0].name)
            movNode = Movl(cnode, vnode)
            assemblyAst.append(movNode)
            
        elif isinstance(node.expr, Name):
            v1node = Var(node.expr.name)
            vnode = Var(node.nodes[0].name)
            movNode = Movl(v1node, vnode)
            assemblyAst.append(movNode)

        elif isinstance(node.expr, FuncName):
            v1node = FuncLabel(node.expr.name)
            vnode = Var(node.nodes[0].name)
            movNode = Movl(v1node, vnode)
            assemblyAst.append(movNode)

        elif isinstance(node.expr, Compare):
            vnode = Var(node.nodes[0].name) 

            r = None
            l = None
            if isinstance(node.expr.expr, Const):
                l = AConst(node.expr.expr.value)
            elif isinstance(node.expr.expr, Name):
                l = Var(node.expr.expr.name)
            else:
                raise Exception("heck")
            if isinstance(node.expr.ops[0][1], Const):
                r = AConst(node.expr.ops[0][1].value)
            elif isinstance(node.expr.ops[0][1], Name):
                r = Var(node.expr.ops[0][1].name)
            else:
                raise Exception("heck")
            
            if isinstance(r, AConst):
                if isinstance(l, Var) or isinstance(l, Reg):
                    l, r = r, l
                elif isinstance(l, AConst):
                    rtmp = Var("cmpTmp")
                    movNode = Movl(r, rtmp)
                    assemblyAst.append(movNode)
                    r = rtmp

            assemblyAst.append(Cmpl(l, r))

            if node.expr.ops[0][0] == "!=":
                assemblyAst.append(SetNe(Reg("al")))
            else:
                assemblyAst.append(SetE(Reg("al")))

            assemblyAst.append(Movzbl(Reg("al"), vnode))

        elif isinstance(node.expr, Add):
            if isinstance(node.expr.left, Const):
                if isinstance(node.expr.right, Const):
                    vnode = Var(node.nodes[0].name)
                    c1node = AConst(node.expr.right.value)
                    movNode = Movl(c1node, vnode)
                    assemblyAst.append(movNode)
                    c2node = AConst(node.expr.left.value)
                    addNode = Addl(c2node, vnode)
                    assemblyAst.append(addNode)
                elif isinstance(node.expr.right, Name):
                    vnode = Var(node.nodes[0].name)
                    if node.expr.right.name != node.nodes[0].name:
                        v1node = Var(node.expr.right.name)
                        movNode = Movl(v1node, vnode)
                        assemblyAst.append(movNode)
                    cnode = AConst(node.expr.left.value)
                    addNode = Addl(cnode, vnode)
                    assemblyAst.append(addNode)
                else:
                    print node.expr.right
                    raise Exception("add is not flat! r1")
            elif isinstance(node.expr.left, Name):
                vnode = Var(node.nodes[0].name)
                if node.expr.left.name != node.nodes[0].name:
                    v1node = Var(node.expr.left.name)
                    movNode = Movl(v1node, vnode)
                    assemblyAst.append(movNode)
                if isinstance(node.expr.right, Const):
                    cnode = AConst(node.expr.right.value)
                    addNode = Addl(cnode, vnode)
                    assemblyAst.append(addNode)
                elif isinstance(node.expr.right, Name):
                    v1node = Var(node.expr.right.name)
                    addNode = Addl(v1node, vnode)
                    assemblyAst.append(addNode)
                else:
                    print node.expr.right
                    raise Exception("add is not flat! r2")
            else:
                raise Exception("add is not flat! l")

        elif isinstance(node.expr, UnarySub):
            if isinstance(node.expr.expr, Const):
                vnode = Var(node.nodes[0].name)
                cnode = AConst(node.expr.expr.value)
                movNode = Movl(cnode, vnode)
                assemblyAst.append(movNode)
                negNode = Negl(vnode)
                assemblyAst.append(negNode)
            elif isinstance(node.expr.expr, Name):
                vnode = Var(node.nodes[0].name)
                if node.expr.expr.name != node.nodes[0].name:
                    v1node = Var(node.expr.expr.name)
                    movNode = Movl(v1node, vnode)
                    assemblyAst.append(movNode)
                negNode = Negl(vnode)
                assemblyAst.append(negNode)
            else:
                print node.expr
                raise Exception("neg is not flat!")

        elif isinstance(node.expr, CallFunc):
            addBack = 0
            for i in range(len(node.expr.args)-1, -1, -1):
                arg = node.expr.args[i]
                pnode = None
                if isinstance(arg, Const):
                    pnode = AConst(arg.value)
                elif isinstance(arg, Name):
                    pnode = Var(arg.name)
                elif isinstance(arg, FuncName):
                    pnode = FuncLabel(arg.name)
                elif isinstance(arg, str):
                    pnode = "${}".format(strings[arg])
                assemblyAst.append(Pushl(pnode))
                addBack += 4
            callNode = Call(node.expr.node.name)
            assemblyAst.append(callNode)

            if addBack:
                addNode = Addl(AConst(addBack), Reg("esp"))
                assemblyAst.append(addNode)

            regNode = Reg("eax")
            vnode = Var(node.nodes[0].name)
            movNode = Movl(regNode,vnode)
            assemblyAst.append(movNode)

        elif isinstance(node.expr, IndCall):
            addBack = 0
            for i in range(len(node.expr.args)-1, -1, -1):
                arg = node.expr.args[i]
                pnode = None
                if isinstance(arg, Const):
                    pnode = AConst(arg.value)
                elif isinstance(arg, Name):
                    pnode = Var(arg.name)
                elif isinstance(arg, FuncName):
                    pnode = FuncLabel(arg.name)
                elif isinstance(arg, str):
                    pnode = "${}".format(strings[arg])
                assemblyAst.append(Pushl(pnode))
                addBack += 4
            if isinstance(node.expr.node, Name):
                callNode = CallInd(Var(node.expr.node.name))
            else:
                raise Exception("bad node type: {}".format(arg))
            assemblyAst.append(callNode)

            if addBack:
                addNode = Addl(AConst(addBack), Reg("esp"))
                assemblyAst.append(addNode)

            regNode = Reg("eax")
            vnode = Var(node.nodes[0].name)
            movNode = Movl(regNode,vnode)
            assemblyAst.append(movNode)

        elif isinstance(node.expr, ProjectTo):
            vnode = Var(node.nodes[0].name)

            pnode = None

            if isinstance(node.expr.arg, Const):
                pnode = AConst(node.expr.arg.value)
            elif isinstance(node.expr.arg, Name):
                pnode = Var(node.expr.arg.name)

            if node.expr.typ == 3:
                assemblyAst.append(Movl(AConst(3), vnode))
                assemblyAst.append(Xorl(AConst(-1), vnode))
                assemblyAst.append(Andl(pnode, vnode))
            else:
                assemblyAst.append(Movl(pnode, vnode))
                assemblyAst.append(Sarl(AConst(2), vnode))

        elif isinstance(node.expr, InjectFrom):
            vnode = Var(node.nodes[0].name)

            pnode = None

            if isinstance(node.expr.arg, Const):
                pnode = AConst(node.expr.arg.value)
            elif isinstance(node.expr.arg, Name):
                pnode = Var(node.expr.arg.name)

            assemblyAst.append(Movl(pnode, vnode))
            if node.expr.typ != 3:
                assemblyAst.append(Shl(AConst(2), vnode))
            assemblyAst.append(Orl(AConst(node.expr.typ), vnode))

        elif isinstance(node.expr, GetTag):
            vnode = Var(node.nodes[0].name)
            
            pnode = None
            if isinstance(node.expr.arg, Const):
                pnode = AConst(node.expr.arg.value)
            elif isinstance(node.expr.arg, Name):
                pnode = Var(node.expr.arg.name)

            assemblyAst.append(Movl(pnode, vnode))
            assemblyAst.append(Andl(AConst(3), vnode))

        else:
            print node
            raise Exception("assignment is no flat")

    elif isinstance(node, While):
        # assembly for test
        testNode = None
        placNode = None
        if isinstance(node.test[1], Name):
            placNode = Var(node.test[1].name)
            testNode = Cmpl(AConst(0), placNode)
        elif isinstance(node.test[1], Const):
            placNode = Var("testTemp")
            movNode = Movl(AConst(node.test[1].value), placNode)
            assemblyAst.append(movNode)
            testNode = Cmpl(AConst(0), placNode)
        else:
            raise Exception("{}".format(node.test))

        testAst = []
        translate_ir(node.test[0], testAst, strings)

        # assembly for body
        bodyAst = []
        translate_ir(node.body, bodyAst, strings)

        testAst.append(testNode)
   
        whileNode = While((testAst, placNode), bodyAst, None)

        assemblyAst.append(whileNode)

    else:
        print node
        raise Exception('unrecognized AST node')

def profiler(node, expected, env):
    if isinstance(node, Module):
        return profiler(node.node, "", env)

    elif isinstance(node, Stmt):
        sd = env.copy()
        typ = ""
        for n in node.nodes:
            if isinstance(n, Assign):
                typ = profiler(n.expr, "", sd)
                if isinstance(n.nodes[0], AssName):
                    sd[n.nodes[0].name] = typ
                else:
                    typ = profiler(n.nodes[0], "", sd)
            else:
                typ = profiler(n, "", sd)
        
        return typ

    elif isinstance(node, Discard):
        typ = profiler(node.expr, "", env)
        return ""

    elif isinstance(node, Printnl):
        typ = profiler(node.nodes[0], "", env)
        if typ != "pyobj":
            raise Exception("type_error: print_any {}".format(typ))
        return ""

    elif isinstance(node, CallFunc):
        """
        for arg in node.args:
            typ = profiler(arg, "", env)
            if typ != "pyobj":
                raise Exception("type error: passing non pyobj argument to {}".format(node))
        """
        return ""

    elif isinstance(node, Const):
        if expected == "bool":
            if node.value == 1 or node.value == 0:
                return "bool"
            raise Exception("type error: not a bool!")

        return "int"

    elif isinstance(node, Name):
        if node.name in env:
            return env[node.name]
        raise Exception("type error: {} not in env!".format(node.name))

    elif isinstance(node, Add):
        tl = profiler(node.left, "int", env)
        tr = profiler(node.right, "int", env)

        if tl == "int" and tr == "int":
            return "int"

        raise Exception("type error: {}: {} + {}: {}".format(node.left, tl, node.right, tr))

    elif isinstance(node, UnarySub):
        te = profiler(node.expr, "int", env)
        
        if te == "int":
            return "int"

        raise Exception("type error: {}: {}".format(node.expr, te))

    elif isinstance(node, IfExp):
        ttest = profiler(node.test, "bool", env)

        if ttest != "bool":
            raise Exception("type error: {}: {}".format(node.test, ttest))

        tthen = profiler(node.then, "", env)
        ethen = profiler(node.else_, "", env)

        if tthen != ethen:
            raise Exception("type error: returns {} or {}, {}".format(tthen, ethen, node))

        return tthen

    elif isinstance(node, InjectFrom):
        rt = profiler(node.arg, "", env)
        if rt == "pyobj":
            raise Exception("type error: InjectFrom(pyobj): {}".format(node))
        return "pyobj"

    elif isinstance(node, ProjectTo):
        rt = profiler(node.arg, "", env)
        if rt != "pyobj":
            raise Exception("type error: ProjectTo(non-pyobj): {}".format(node))

        if node.typ == 0:
            return "int"
        elif node.typ == 1:
            return "bool"
        elif node.typ == 3:
            return "big"
        
        raise Exception("type error: unrecognized projection {}".format(node.typ))

    elif isinstance(node, GetTag):
        at = profiler(node.arg, "", env)
        if at != "pyobj":
            raise Exception("type error: arg is not pyobj {}: {}".format(node, at))

        return "int"

    elif isinstance(node, Compare):
        lt = profiler(node.expr, "", env)
        rt = profiler(node.ops[0][1], "", env)

        if node.ops[0][0] == "is" or node.ops[0][0] == "is not":
            if lt == "pyobj" and rt == "pyobj":
                return "bool"
            else:
                raise Exception("type error: Compare: {} is {}".format(lt, rt))
        else:
            if (lt == "bool" or lt == "int") and (rt == "bool" or rt == "int"):
                return "bool"

        raise Exception("type error: Compare {}, {}".format(lt, rt))

    elif isinstance(node, Let):
        tsub = profiler(node.sub, "", env)
        nd = {k: env[k] for k in env}
        nd[node.var.name] = tsub
        texpr = profiler(node.expr, "", nd)

        return texpr

    elif isinstance(node, Subscript):
        tl = profiler(node.expr , "", env)
        if tl != "pyobj":
            raise Exception("type error: Subscript({}, _)".format(tl))

        tr = profiler(node.subs[0] , "", env)
        if tr != "pyobj":
            raise Exception("type error: Subscript({}, {})".format(tl, tr))

        return "pyobj"

    return ""

def gather_assignments(n):
    if isinstance(n, Stmt):
        s = [gather_assignments(e) for e in n.nodes]
        return set(reduce(lambda a,b: a|b, s, set([])))
    elif isinstance(n, Function):
        return set([n.name])
    elif isinstance(n, Assign):
        if isinstance(n.nodes[0], AssName):
            return set([n.nodes[0].name])
        else:
            return set([])
    elif isinstance(n, Let):
        return set([n.var.name])
    elif isinstance(n, Class):
        return set([n.name])
    elif isinstance(n, If):
        st = gather_assignments(n.tests[0][1])
        se = gather_assignments(n.else_)
        return st|se

    elif isinstance(n, While):
        return gather_assignments(n.body)

    return set([])

def uniquify(node, u_env, index):
    if isinstance(node, Module):
        statements, index = uniquify(node.node, u_env, index)
        return Module("", statements), index

    elif isinstance(node, InjectFrom):
        u_a, index = uniquify(node.arg, u_env, index)
        return InjectFrom(node.typ, u_a), index

    elif isinstance(node, Stmt):
        a_vars = gather_assignments(node)
        nu_env = {}
        for v in a_vars:
            nu_env[v] = "{}_{}".format(v, index)
            index += 1
        for k in u_env:
            if k not in nu_env:
                nu_env[k] = u_env[k]

        statements = []
        for n in node.nodes:
            u_n, index = uniquify(n, nu_env, index)
            statements.append(u_n)
        return Stmt(statements), index

    elif isinstance(node, Let):
        u_s, index = uniquify(node.sub, u_env, index)
        nu_env = dict(u_env)
        nu_env[node.var] = node.var
        u_e, index = uniquify(node.expr, nu_env, index)
        return Let(node.var, u_s, u_e), index

    elif isinstance(node, str):
        return node, index

    elif isinstance(node, Function):
        new_env = dict(u_env)
        new_function_name = u_env[node.name]
        new_arg_names = []
        for n in node.argnames:
            new_env[n] = "{}_{}".format(n, index)
            index += 1
            new_arg_names.append(new_env[n])
        
        a_vars = gather_assignments(node.code)
        for v in a_vars:
            if v not in node.argnames:
                new_env[v] = "{}_{}".format(v, index)
                index += 1

        for k in u_env:
            if k not in new_env:
                new_env[k] = u_env[k]

        code = []
        for n in node.code.nodes:
            u_n, index = uniquify(n, new_env, index)
            code.append(u_n)
        new_code = Stmt(code)
        #new_code, index = uniquify(node.code, new_env, index)
        return Function(node.decorators, new_function_name, new_arg_names, node.defaults, node.flags, node.doc, new_code), index

    elif isinstance(node, Lambda):
        new_env = {}

        new_arg_names = []
        for n in node.argnames:
            new_env[n] = "{}_{}".format(n, index)
            index += 1
            new_arg_names.append(new_env[n])

        for k in u_env:
            if k not in new_env:
                new_env[k] = u_env[k]

        new_code, index = uniquify(node.code, new_env, index)

        return Lambda(new_arg_names, node.defaults, node.flags, new_code), index

    elif isinstance(node, Return):
        new_value, index = uniquify(node.value, u_env, index)
        return Return(new_value), index

    elif isinstance(node, Discard):
        new_expr, index = uniquify(node.expr, u_env, index)
        return Discard(new_expr), index

    elif isinstance(node, Printnl):
        new_node, index = uniquify(node.nodes[0], u_env, index)
        return Printnl([new_node], None), index

    elif isinstance(node, Assign):
        new_expr, index = uniquify(node.expr, u_env, index)
        new_left, index = uniquify(node.nodes[0], u_env, index)
        return Assign([new_left], new_expr), index

    elif isinstance(node, AssName):
        if node.name == "input" or node.name == "True" or node.name == "False" or node.name in reservedFunctions:
            return node, index
        if node.name not in u_env:
            return node, index
        return AssName(u_env[node.name], None), index

    elif isinstance(node, CallFunc):
        new_node, index = uniquify(node.node, u_env, index)
        new_args = []
        for n in node.args:
            new_n, index = uniquify(n, u_env, index)
            new_args.append(new_n)
        return CallFunc(new_node, new_args), index 

    elif isinstance(node, IfExp):
        new_test, index = uniquify(node.test, u_env, index)
        new_then, index = uniquify(node.then, u_env, index)
        new_else_, index = uniquify(node.else_, u_env, index)

        return IfExp(new_test, new_then, new_else_), index

    elif isinstance(node, Compare):
        new_left, index = uniquify(node.expr, u_env, index)
        new_right, index = uniquify(node.ops[0][1], u_env, index)
        new_op = (node.ops[0][0], new_right)
        return Compare(new_left, [new_op]), index

    elif isinstance(node, Add):
        new_left, index = uniquify(node.left, u_env, index)
        new_right, index = uniquify(node.right, u_env, index)
        return Add((new_left, new_right)), index

    elif isinstance(node, And):
        new_left, index = uniquify(node.nodes[0], u_env, index)
        new_right, index = uniquify(node.nodes[1], u_env, index)
        return And([new_left, new_right]), index

    elif isinstance(node, Or):
        new_left, index = uniquify(node.nodes[0], u_env, index)
        new_right, index = uniquify(node.nodes[1], u_env, index)
        return Or([new_left, new_right]), index

    elif isinstance(node, Not):
        new_expr, index = uniquify(node.expr, u_env, index)
        return Not(new_expr), index

    elif isinstance(node, UnarySub):
        new_expr, index = uniquify(node.expr, u_env, index)
        return UnarySub(new_expr), index

    elif isinstance(node, Subscript):
        new_expr, index = uniquify(node.expr, u_env, index)
        new_arg, index = uniquify(node.subs[0], u_env, index)
        return Subscript(new_expr, node.flags, [new_arg]), index

    elif isinstance(node, List):
        new_nodes = []
        for n in node:
            new_n, index = uniquify(n, u_env, index)
            new_nodes.append(new_n)
        return List(new_nodes), index

    elif isinstance(node, Dict):
        new_items = []
        for k,v in node.items:
            n_k, index = uniquify(k, u_env, index)
            n_v, index = uniquify(v, u_env, index)
            new_items.append((n_k, n_v))
        return Dict(new_items), index

    elif isinstance(node, Const):
        return node, index

    elif isinstance(node, Name):
        if node.name == "input" or node.name == "True" or node.name == "False" or node.name in reservedFunctions:
            return node, index
        if node.name not in u_env:
            return node, index
        return Name(u_env[node.name]), index

    elif isinstance(node, If):
        new_test, index = uniquify(node.tests[0][0], u_env, index)

        then_code = []
        for n in node.tests[0][1].nodes:
            u_n, index = uniquify(n, u_env, index)
            then_code.append(u_n)
        new_then = Stmt(then_code)

        else_code = []
        for n in node.else_.nodes:
            u_n, index = uniquify(n, u_env, index)
            else_code.append(u_n)
        new_else_ = Stmt(else_code)

        return If([(new_test, new_then)], new_else_), index

    elif isinstance(node, While):
        new_test, index = uniquify(node.test, u_env, index)
        #new_body, index = uniquify(node.body, u_env, index)

        body_code = []
        for n in node.body.nodes:
            u_n, index = uniquify(n, u_env, index)
            body_code.append(u_n)
        new_body = Stmt(body_code)

        return While(new_test, new_body, node.else_), index

    raise Exception("unrecognized ast node in uniquify {}".format(node))

def declassify(node, index, parent, top_level, assignments, functionMap):
    if isinstance(node, Module):
        dn, index = declassify(node.node, index, parent, top_level, assignments, functionMap)
        
        return Module("", dn)

    elif isinstance(node, Stmt):
        statements = []
        for n in node.nodes:
            dn, index = declassify(n, index, parent, top_level, assignments, functionMap)
            if isinstance(dn, Stmt):
                for i in dn.nodes:
                    statements.append(i)
            else:
                statements.append(dn)

        return Stmt(statements), index

    elif isinstance(node, Class):
        newParent = "{}class".format(index)
        index += 1

        classNode = Assign([AssName(newParent, None)], InjectFrom(3, CallFunc(Name("create_class"), [List([x for x in node.bases])])))

        a_vars = gather_assignments(node.code)
        a_vars = a_vars - reservedFunctions

        newbody, index = declassify(node.code, index, newParent, top_level, a_vars, functionMap)

        assNode = None
        if not parent:
            assNode = Assign([AssName(node.name, None)], Name(newParent))
        else:
            assNode = CallFunc(Name("set_attr"), [Name(parent), node.name, Name(newParent)])

        return Stmt([classNode]+newbody.nodes+[assNode]), index

    elif isinstance(node, AssAttr):
        lhs, index = declassify(node.expr, index, parent, top_level, assignments, functionMap) 
        return AssAttr(lhs, node.attrname, node.flags), index

    elif isinstance(node, Getattr):
        lhs, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        return CallFunc(Name("get_attr"), [lhs, node.attrname]), index

    elif isinstance(node, If):
        dt, index = declassify(node.tests[0][0], index, parent, top_level, assignments, functionMap)
        dtt, index = declassify(node.tests[0][1], index, parent, top_level, assignments, functionMap)
        de, index = declassify(node.else_, index, parent, top_level, assignments, functionMap)
        return If([(dt, dtt)], de), index

    elif isinstance(node, While):
        dt, index = declassify(node.test, index, parent, top_level, assignments, functionMap)
        db, index = declassify(node.body, index, parent, top_level, assignments, functionMap)
        return While(dt, db, node.else_), index

    elif isinstance(node, Function):
        if parent:
            newFuncName = "{}_{}".format(node.name, index)
            index += 1
            extendedMap = dict(functionMap)
            extendedMap[node.name] = newFuncName
            dcode, index = declassify(node.code, index, None, top_level, set([]), extendedMap)
            newFunc = Function(None, newFuncName, node.argnames, node.defaults, node.flags, "", dcode)
            setCall = CallFunc(Name("set_attr"), [Name(parent), node.name, Name(newFuncName)])
            return Stmt([newFunc, setCall]), index

        return node, index

    elif isinstance(node, Lambda):
        dc, index = declassify(node.code, index, None, top_level, set([]), functionMap)
        return Lambda(node.argnames, node.defaults, node.flags, dc), index

    elif isinstance(node, Return):
        de, index = declassify(node.value, index, parent, top_level, assignments, functionMap)
        return Return(de), index

    elif isinstance(node, Discard):
        dc, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        return Discard(dc), index

    elif isinstance(node, Printnl):
        dn, index = declassify(node.nodes[0], index, parent, top_level, assignments, functionMap)
        return Printnl([dn], node.dest), index

    elif isinstance(node, Assign):
        lhs, index = declassify(node.nodes[0], index, parent, top_level, assignments, functionMap)
        rhs, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        if isinstance(lhs, AssAttr):
            return CallFunc(Name("set_attr"), [lhs.expr, lhs.attrname, rhs]), index
        return Assign([lhs], rhs), index

    elif isinstance(node, CallFunc):
        dn, index = declassify(node.node, index, parent, top_level, assignments, functionMap)
        des = []
        for e in node.args:
            de, index = declassify(e, index, parent, top_level, assignments, functionMap)
            des.append(de)
        if isinstance(dn, Name) and dn.name in reservedFunctions:
            return CallFunc(dn, des), index

        argnames = []
        for i in range(len(des)):
            argnames.append("{}arg".format(i))

        b1exp = Let(Name("_"), CallFunc(Name("1234ini"), [Name("1234o")]+[Name(ae) for ae in argnames]), Name("1234o"))
        b2exp = Let(Name("1234ini"), InjectFrom(3, CallFunc(Name("get_function"), [CallFunc(Name("get_attr"), [Name("1234f"), '__init__'])])), b1exp)
        iexp = IfExp(
            InjectFrom(1, CallFunc(Name("has_attr"), [Name("1234f"), '__init__'])),
            b2exp,
            Name("1234o"))
        texp = Let(Name("1234o"), InjectFrom(3, CallFunc(Name("create_object"), [Name("1234f")])), iexp)

        fexp = IfExp(
            InjectFrom(1, CallFunc(Name("is_bound_method"), [Name("1234f")])),
            Let(
                Name("1rarg"), 
                InjectFrom(3, CallFunc(Name("get_receiver"), [Name("1234f")])), 
                CallFunc(
                    InjectFrom(3, CallFunc(Name("get_function"), [Name("1234f")])),
                    [Name("1rarg")] + [Name(ae) for ae in argnames]
                )
            ),
            IfExp(
                InjectFrom(1, CallFunc(Name("is_unbound_method"), [Name("1234f")])),
                CallFunc(
                    InjectFrom(3, CallFunc(Name("get_function"), [Name("1234f")])),
                    [Name(ae) for ae in argnames]
                ),
                CallFunc(Name("1234f"), [Name(ae) for ae in argnames])
            )
        )

        lexp = IfExp(
            InjectFrom(1, CallFunc(Name("is_class"), [Name("1234f")])),
            texp,
            fexp
        )
            

        for i in range(len(des)-1, -1, -1):
            lexp = Let(Name("{}arg".format(i)), des[i], lexp)

        lexp = Let(Name("1234f"), dn, lexp)
        
        return lexp, index
        

    elif isinstance(node, IfExp):
        dt, index = declassify(node.test, index, parent, top_level, assignments, functionMap)
        dtt, index = declassify(node.then, index, parent, top_level, assignments, functionMap)
        de, index = declassify(node.else_, index, parent, top_level, assignments, functionMap)
        return IfExp(dt, dtt, de), index

    elif isinstance(node, Compare):
        dl, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        dr, index = declassify(node.ops[0][1], index, parent, top_level, assignments, functionMap)
        return Compare(dl, [(node.ops[0][0], dr)]), index

    elif isinstance(node, Add):
        dl, index = declassify(node.left, index, parent, top_level, assignments, functionMap)
        dr, index = declassify(node.right, index, parent, top_level, assignments, functionMap)
        return Add((dl, dr)), index

    elif isinstance(node, And):
        dl, index = declassify(node.nodes[0], index, parent, top_level, assignments, functionMap)
        dr, index = declassify(node.nodes[1], index, parent, top_level, assignments, functionMap)
        return And([dl, dr]), index

    elif isinstance(node, Or):
        dl, index = declassify(node.nodes[0], index, parent, top_level, assignments, functionMap)
        dr, index = declassify(node.nodes[1], index, parent, top_level, assignments, functionMap)
        return Or([dl, dr]), index

    elif isinstance(node, Not):
        de, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        return Not(de), index

    elif isinstance(node, UnarySub):
        de, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        return UnarySub(de), index

    elif isinstance(node, Subscript):
        e, index = declassify(node.expr, index, parent, top_level, assignments, functionMap)
        arg, index = declassify(node.subs[0], index, parent, top_level, assignments, functionMap)
        return Subscript(e, node.flags, [arg]), index

    elif isinstance(node, List):
        elems = []
        for n in node.nodes:
            dn, index = declassify(n, index, parent, top_level, assignments, functionMap)
            elems.append(dn)

        return List(elems), index

    elif isinstance(node, Dict):
        elems = []
        for k, v in node.items:
            dk, index = declassify(k, index, parent, top_level, assignments, functionMap)
            dv, index = declassify(v, index, parent, top_level, assignments, functionMap)
            elems.append((dk, dv))

        return Dict(elems), index

    elif isinstance(node, Const):
        return node, index

    elif isinstance(node, Name):
        if node.name in assignments and parent:
            if node.name in top_level:
                return IfExp(
                    InjectFrom(1, CallFunc(Name("has_attr"), [Name(parent), node.name])),
                    CallFunc(Name("get_attr"), [Name(parent), node.name]),
                    node
                ), index
            else:
                #return Getattr(Name(parent), node.name), index
                return CallFunc(Name("get_attr"), [Name(parent), node.name]), index
        elif node.name in functionMap:
            return Name(functionMap[node.name]), index
        return node, index
    
    elif isinstance(node, AssName):
        if node.name in assignments and parent:
            attrNode = AssAttr(Name(parent), node.name, 'OP_ASSIGN')
            return attrNode, index
        return node, index

    else:
        raise Exception("unrecognized ast node in declassify! {}".format(node))

def explicate(node):
    if isinstance(node, Module):
        statements = explicate(node.node)
        return Module("", statements)

    elif isinstance(node, str):
        return node

    elif isinstance(node, InjectFrom):
        exp_a = explicate(node.arg)
        return InjectFrom(node.typ, exp_a)

    elif isinstance(node, Stmt):
        statements = []
        for n in node.nodes:
            expression = explicate(n)
            statements.append(expression)
        return Stmt(statements)

    elif isinstance(node, Let):
        exp_sub = explicate(node.sub)
        exp_e = explicate(node.expr)
        return Let(node.var, exp_sub, exp_e)

    elif isinstance(node, Function):
        lhs = AssName(node.name, None)
        exp_code = explicate(node.code)
        lamb = Lambda(node.argnames, node.defaults, node.flags, exp_code)
        assExp = Assign([lhs], lamb)
        return assExp

    elif isinstance(node, Lambda):
        exp_code = explicate(node.code)
        new_body = Stmt([Return(exp_code)])
        return Lambda(node.argnames, node.defaults, node.flags, new_body)

    elif isinstance(node, Return):
        if isinstance(node.value, Const):
            if node.value.value is None:
                exp_e = InjectFrom(0, Const(0))
            else:
                exp_e = explicate(node.value)
        else:
            exp_e = explicate(node.value)
        return Return(exp_e)

    elif isinstance(node, Discard):
        return Discard(explicate(node.expr))

    elif isinstance(node, Printnl):
        printExp = Printnl([Name("1234printarg")], None)
        printLet = Let(Name("1234printarg"), explicate(node.nodes[0]), printExp)
        return printLet

    elif isinstance(node, Assign):
        lhs = node.nodes[0] if isinstance(node.nodes[0], AssName) else explicate(node.nodes[0])
        assExp = Assign([lhs], explicate(node.expr))
        return assExp

    elif isinstance(node, CallFunc):
        if isinstance(node.node, Name):
            if node.node.name == "input":
                return InjectFrom(0, node)
        expArgs = [explicate(a) for a in node.args]
        expNode = explicate(node.node)
        return CallFunc(expNode, expArgs)

    elif isinstance(node, IfExp):
        expTest = IfExp(
            Compare(GetTag(Name("1234test")), [("==", Const(1))]),
            Name("1234test"),
            IfExp(
                Compare(GetTag(Name("1234test")), [("==", Const(0))]),
                InjectFrom(1, Compare(ProjectTo(0, Name("1234test")), [("!=", Const(0))])),
                IfExp(
                    Compare(GetTag(Name("1234test")), [("==", Const(3))]),
                    InjectFrom(1, CallFunc(Name("is_true"), [Name("1234test")])),
                    InjectFrom(1, CallFunc(Name("exit"), []))
                )
            )
        )
        letTest = ProjectTo(1, Let(Name("1234test"), explicate(node.test), expTest))
                
        expThen = explicate(node.then)
        expElse = explicate(node.else_)
        return IfExp(letTest, expThen, expElse)

    elif isinstance(node, Compare):
        if node.ops[0][0] == "is":
            compExp = InjectFrom(1, Compare(Name("1234left"), [(node.ops[0][0], Name("1234right"))]))
        elif node.ops[0][0] == "is not":
            compExp = InjectFrom(1, 
                Compare(
                    Compare(Name("1234left"), [(node.ops[0][0], Name("1234right"))]),
                    [
                        ("==", Const(0))
                    ]
                )
            )
        else:
            bigCall = InjectFrom(1, CallFunc(Name("equal"), [ProjectTo(3, Name("1234left")), ProjectTo(3, Name("1234right"))]))
            if node.ops[0][0] == "!=":
                bigCall = InjectFrom(1, CallFunc(Name("not_equal"), [ProjectTo(3, Name("1234left")), ProjectTo(3, Name("1234right"))]))
            compExp = IfExp(
                IfExp(
                    Compare(GetTag(Name("1234left")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    IfExp(
                        Compare(GetTag(Name("1234left")), [("==", Const(1))]),
                        ProjectTo(1, InjectFrom(1, Const(1))),
                        ProjectTo(1, InjectFrom(1, Const(0))),
                    )
                ),
                IfExp(
                    IfExp(
                        Compare(GetTag(Name("1234right")), [("==", Const(0))]),
                        ProjectTo(1, InjectFrom(1, Const(1))),
                        IfExp(
                            Compare(GetTag(Name("1234right")), [("==", Const(1))]),
                            ProjectTo(1, InjectFrom(1, Const(1))),
                            ProjectTo(1, InjectFrom(1, Const(0))),
                        )
                    ),
                    InjectFrom(1, Compare(ProjectTo(0, Name("1234left")), [(node.ops[0][0], ProjectTo(0, Name("1234right")))])),
                    InjectFrom(1, Const(0)) if node.ops[0][0] == "==" else InjectFrom(1, Const(1))
                ),
                IfExp(
                    Compare(GetTag(Name("1234right")), [("==", Const(3))]),
                    bigCall,
                    InjectFrom(1, Const(0)) if node.ops[0][0] == "==" else InjectFrom(1, Const(1))
                )
            )
        rightLet = Let(Name("1234right"), explicate(node.ops[0][1]), compExp)
        leftLet = Let(Name("1234left"), explicate(node.expr), rightLet)
        return leftLet

    elif isinstance(node, Add):
        addExp = IfExp(
            IfExp(
                Compare(GetTag(Name("1234left")), [("==", Const(0))]),
                IfExp(
                    Compare(GetTag(Name("1234right")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    IfExp(
                        Compare(GetTag(Name("1234right")), [("==", Const(1))]),
                        ProjectTo(1, InjectFrom(1, Const(1))),
                        ProjectTo(1, InjectFrom(1, Const(0)))
                    )
                ),
                IfExp(
                    Compare(GetTag(Name("1234left")), [("==", Const(1))]),
                    IfExp(
                        Compare(GetTag(Name("1234right")), [("==", Const(0))]),
                        ProjectTo(1, InjectFrom(1, Const(1))),
                        IfExp(
                            Compare(GetTag(Name("1234right")), [("==", Const(1))]),
                            ProjectTo(1, InjectFrom(1, Const(1))),
                            ProjectTo(1, InjectFrom(1, Const(0)))
                        )
                    ),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                )
            ),    
            InjectFrom(0, Add((ProjectTo(0, Name("1234left")), ProjectTo(0, Name("1234right"))))),
            IfExp(
                IfExp(
                    Compare(GetTag(Name("1234left")), [("==", Const(3))]),
                    IfExp(
                        Compare(GetTag(Name("1234right")), [("==", Const(3))]),
                        ProjectTo(1, InjectFrom(1, Const(1))),
                        ProjectTo(1, InjectFrom(1, Const(0)))
                    ),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                ),
                InjectFrom(3, CallFunc(Name("add"), [ProjectTo(3, Name("1234left")), ProjectTo(3, Name("1234right"))])),
                InjectFrom(3, CallFunc(Name("exit"), []))
            )
        )
        rightLet = Let(Name("1234right"), explicate(node.right), addExp)
        leftLet = Let(Name("1234left"), explicate(node.left), rightLet)

        return leftLet

    elif isinstance(node, And):
        andExp = IfExp(
            IfExp(
                Compare(GetTag(Name("1234left")), [("==", Const(1))]),
                ProjectTo(1, InjectFrom(1, Const(1))),
                IfExp(
                    Compare(GetTag(Name("1234left")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                )
            ),
            IfExp(
                Compare(ProjectTo(0, Name("1234left")),[('==', Const(0))]),
                Name("1234left"),
                explicate(node.nodes[1])
            ),
            IfExp(
                Compare(GetTag(Name("1234left")), [("==", Const(3))]),
                IfExp(
                    ProjectTo(1, InjectFrom(1, CallFunc(Name("is_true"), [Name("1234left")]))),
                    explicate(node.nodes[1]),
                    Name("1234left")
                ),  
                InjectFrom(1, CallFunc(Name("exit"), []))
            )
        )
        leftLet = Let(Name("1234left"), explicate(node.nodes[0]), andExp)

        return leftLet

    elif isinstance(node, Or):
        orExp = IfExp(
            IfExp(
                Compare(GetTag(Name("1234left")), [("==", Const(1))]),
                ProjectTo(1, InjectFrom(1, Const(1))),
                IfExp(
                    Compare(GetTag(Name("1234left")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                )
            ),
            IfExp(
                Compare(ProjectTo(0, Name("1234left")),[('==', Const(0))]),
                explicate(node.nodes[1]),
                Name("1234left")
            ),
            IfExp(
                Compare(GetTag(Name("1234left")), [("==", Const(3))]),
                IfExp(
                    ProjectTo(1, InjectFrom(1, CallFunc(Name("is_true"), [Name("1234left")]))),
                    Name("1234left"),
                    explicate(node.nodes[1])
                ),  
                InjectFrom(1, CallFunc(Name("exit"), []))
            )
        )
        leftLet = Let(Name("1234left"), explicate(node.nodes[0]), orExp)

        return leftLet

    elif isinstance(node, Not):
        notExp = IfExp(
            IfExp(
                Compare(GetTag(Name("1234arg")), [("==", Const(1))]),
                ProjectTo(1, InjectFrom(1, Const(1))),
                IfExp(
                    Compare(GetTag(Name("1234arg")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                )
            ),
            InjectFrom(1, Compare(ProjectTo(0, Name("1234arg")), [("==", Const(0))])),
            IfExp(
                Compare(GetTag(Name("1234arg")), [("==", Const(3))]),
                InjectFrom(1, Compare(ProjectTo(1, InjectFrom(1, CallFunc(Name("is_true"), [Name("1234arg")]))), [("==", Const(0))])),
                InjectFrom(1, CallFunc(Name("exit"), []))
            )
        )
        notLet = Let(Name("1234arg"), explicate(node.expr), notExp)

        return notLet

    elif isinstance(node, UnarySub):
        unaryExp = IfExp(
            IfExp(
                Compare(GetTag(Name("1234arg")), [("==", Const(1))]),
                ProjectTo(1, InjectFrom(1, Const(1))),
                IfExp(
                    Compare(GetTag(Name("1234arg")), [("==", Const(0))]),
                    ProjectTo(1, InjectFrom(1, Const(1))),
                    ProjectTo(1, InjectFrom(1, Const(0)))
                )
            ),
            InjectFrom(0, UnarySub(ProjectTo(0, Name("1234arg")))),
            InjectFrom(1, CallFunc(Name("exit"), []))
        )
        unaryLet = Let(Name("1234arg"), explicate(node.expr), unaryExp)
        
        return unaryLet

    elif isinstance(node, Subscript):
        e = explicate(node.expr)
        arg = explicate(node.subs[0])
        newSub = Subscript(e, node.flags, [arg])
        return newSub

    elif isinstance(node, List):
        listCreate = InjectFrom(3, CallFunc(Name("create_list"), [InjectFrom(0, Const(len(node.nodes)))]))
        
        # if list is empty, return creation
        if not len(node.nodes):
            return listCreate

        taggedL = [explicate(n) for n in node.nodes]
        
        # 1234list is an invalid variable name, therefore it's safe to use as a temp variable
        cexpr = Name("1234list")
        for i in range(len(taggedL)-1, -1, -1):
            cexpr = Let(
                Name("1234set"),
                CallFunc(Name("set_subscript"), [Name("1234list"), InjectFrom(0, Const(i)), taggedL[i]]),
                cexpr
            )
        creationLet = Let(Name("1234list"), listCreate, cexpr)

        return creationLet

    elif isinstance(node, Dict):
        dictCreate = InjectFrom(3, CallFunc(Name("create_dict"), []))
        if not node.items:
            return dictCreate

        taggedItems = [(explicate(k), explicate(v)) for k,v in node.items]

        cexpr = Name("1234dict")
        for i in range(len(taggedItems)-1, -1, -1):
            cexpr = Let(
                Name("1234set"),
                CallFunc(Name("set_subscript"), [Name("1234dict"), taggedItems[i][0], taggedItems[i][1]]),
                cexpr
            )
        creationLet = Let(Name("1234dict"), dictCreate, cexpr)

        return creationLet

    elif isinstance(node, Const):
        return InjectFrom(0, node)

    elif isinstance(node, Name):
        if node.name == "True":
            return InjectFrom(1, Const(1))
        elif node.name == "False":
            return InjectFrom(1, Const(0))
        return node

    elif isinstance(node, If):
        expTest = IfExp(
            Compare(GetTag(Name("1234test")), [("==", Const(1))]),
            Name("1234test"),
            IfExp(
                Compare(GetTag(Name("1234test")), [("==", Const(0))]),
                InjectFrom(1, Compare(ProjectTo(0, Name("1234test")), [("!=", Const(0))])),
                IfExp(
                    Compare(GetTag(Name("1234test")), [("==", Const(3))]),
                    InjectFrom(1, CallFunc(Name("is_true"), [Name("1234test")])),
                    InjectFrom(1, CallFunc(Name("exit"), []))
                )
            )
        )
        letTest = ProjectTo(1, Let(Name("1234test"), explicate(node.tests[0][0]), expTest))
                
        expThen = explicate(node.tests[0][1])
        expElse = explicate(node.else_)
        return If([(letTest, expThen)], expElse)

    elif isinstance(node, While):
        expTest = IfExp(
            Compare(GetTag(Name("1234test")), [("==", Const(1))]),
            Name("1234test"),
            IfExp(
                Compare(GetTag(Name("1234test")), [("==", Const(0))]),
                InjectFrom(1, Compare(ProjectTo(0, Name("1234test")), [("!=", Const(0))])),
                IfExp(
                    Compare(GetTag(Name("1234test")), [("==", Const(3))]),
                    InjectFrom(1, CallFunc(Name("is_true"), [Name("1234test")])),
                    InjectFrom(1, CallFunc(Name("exit"), []))
                )
            )
        )
        letTest = ProjectTo(1, Let(Name("1234test"), explicate(node.test), expTest))
        
        expBody = explicate(node.body)

        return While(letTest, expBody, node.else_)
    else:
        raise Exception("unrecognized ast node in explicate! {}".format(node))

def flatten(node, index, varMap, strings):
    if isinstance(node, Module):
        index, statements = flatten(node.node, index, varMap, strings)
        return (index, Module("", statements))

    elif isinstance(node, Stmt):
        statementList = []
        for n in node.nodes:
            blah = flatten(n, index, varMap, strings)
            (index, expression, lastVar) = blah
            statementList.append(expression)

        flattenedList = [elem for row in statementList for elem in row]
        
        return (index, Stmt(flattenedList))

    elif isinstance(node, str):
        strings[node] = 1

        return (index, [], node)

    elif isinstance(node, Printnl):
        if len(node.nodes) != 1:
            raise Exception("trying to print more than one thing")

        index, exprs, n = flatten(node.nodes[0], index, varMap, strings)
        newPrintNode = Printnl([n], None)
        exprs.append(newPrintNode)
        return (index, exprs, None)
        

    elif isinstance(node, Assign):
        if len(node.nodes) != 1:
            raise Exception('number of variables to assign is not 1!')
    
        lhs = node.nodes[0]
        lhsExprs = []
        if isinstance(lhs, Subscript):
            index, lse, ls = flatten(lhs.expr, index, varMap, strings)
            index, rse, rs = flatten(lhs.subs[0], index, varMap, strings)
            index, exprs, n = flatten(node.expr, index, varMap, strings)
            lhs = CallFunc(Name("set_subscript"), [ls, rs, n])
            return (index, exprs+lse+rse+[lhs], None)

        index, exprs, n = flatten(node.expr, index, varMap, strings)
        newTemp = ""
        if isinstance(lhs, AssName):
            if lhs.name in varMap:
                newTemp = varMap[lhs.name]
            else:
                newTemp = "t{}".format(index)
                varMap[lhs.name] = newTemp
                index += 1
            lhs.name = newTemp
        newAssNode = Assign([lhs], n)
        exprs.append(newAssNode)
        return (index, exprs, None)

    elif isinstance(node, Discard):
        if isinstance(node.expr, Const):
            if node.expr.value is None:
                return (index, [], None)
            newAssNode = Assign([AssName("t{}".format(index), None)], node.expr)
            index += 1
            return (index, [newAssNode], None)
        elif isinstance(node.expr, InjectFrom) and isinstance(node.expr.arg, Const) and node.expr.arg.value is None:
            return (index, [], None)
        elif isinstance(node.expr, Name):
            if node.expr.name not in varMap:
                raise Exception('referencing variable {} before declaration'.format(node.expr.name))
            node.expr.name = varMap[node.expr.name]
            newAssNode = Assign([AssName("t{}".format(index), None)], node.expr)
            index += 1
            return (index, [newAssNode], None)
        else:
            return flatten(node.expr, index, varMap, strings)

    elif isinstance(node, Subscript):
        index, lhsExprs, lhs = flatten(node.expr, index, varMap, strings)
        index, rhsExprs, rhs = flatten(node.subs[0], index, varMap, strings)
        newTemp = "t{}".format(index)
        newAssNode = Assign([AssName(newTemp, None)], CallFunc(Name("get_subscript"), [lhs, rhs]))
        index += 1
        return (index, lhsExprs+rhsExprs+[newAssNode], Name(newTemp))

    elif isinstance(node, IfExp):
        index, testExprs, test = flatten(node.test, index, varMap, strings)
        index, thenExprs, then = flatten(node.then, index, varMap, strings)
        index, elseExprs, els = flatten(node.else_, index, varMap, strings)

        newTemp = "t{}".format(index)
        index += 1

        thenAss = Assign([AssName(newTemp, None)], then)
        thenExprs.append(thenAss)

        elseAss = Assign([AssName(newTemp, None)], els)
        elseExprs.append(elseAss)

        ifTransform = IfA(test, Stmt(thenExprs), Stmt(elseExprs))

        testExprs.append(ifTransform)

        return (index, testExprs, Name(newTemp))

    elif isinstance(node, If):
        index, testExprs, test = flatten(node.tests[0][0], index, varMap, strings)
        index, thenExprs  = flatten(node.tests[0][1], index, varMap, strings)
        index, elseExprs  = flatten(node.else_, index, varMap, strings)

        ifTransform = IfA(test, thenExprs, elseExprs)

        testExprs.append(ifTransform)

        return (index, testExprs, Name("unused"))

    elif isinstance(node, While):
        index, testExprs, test = flatten(node.test, index, varMap, strings)
        index, bodyExprs = flatten(node.body, index, varMap, strings)

        whileTransform = While((Stmt(testExprs), test), bodyExprs, node.else_)

        return (index, [whileTransform], Name("unused"))

    elif isinstance(node, Add):
        index, exprsLeft, leftNode = flatten(node.left, index, varMap, strings)
        index, exprsRight, rightNode = flatten(node.right, index, varMap, strings)
        newTemp = "t{}".format(index)
        newAssNode = Assign([AssName(newTemp, None)], Add((leftNode, rightNode)))
        index += 1
        exprs = exprsLeft + exprsRight + [newAssNode]
        return (index, exprs, Name(newTemp))

    elif isinstance(node, UnarySub):
        index, exprs, n = flatten(node.expr, index, varMap, strings)
        newTemp = "t{}".format(index)

        newAssNode = Assign([AssName(newTemp, None)], UnarySub(n))
        index += 1
        exprs.append(newAssNode)
        return (index, exprs, Name(newTemp))

    elif isinstance(node, CallFunc):
        argExprs = []
        fargs = []
        for i in range(len(node.args)-1, -1, -1):
            arg = node.args[i]
            index, exprs, n = flatten(arg, index, varMap, strings)
            fargs = [n] + fargs
            argExprs += exprs
        newTemp = "t{}".format(index)
        index, nexprs, new_node = flatten(node.node, index, varMap, strings)
        newAssNode = Assign([AssName(newTemp, None)], CallFunc(new_node, fargs))
        index += 1
        return (index, argExprs + nexprs + [newAssNode], Name(newTemp))

    elif isinstance(node, IndCall):
        argExprs = []
        fargs = []
        for i in range(len(node.args)-1, -1, -1):
            arg = node.args[i]
            index, exprs, n = flatten(arg, index, varMap, strings)
            fargs = [n] + fargs
            argExprs += exprs
        newTemp = "t{}".format(index)
        index, nexprs, new_node = flatten(node.node, index, varMap, strings)
        newAssNode = Assign([AssName(newTemp, None)], IndCall(new_node, fargs))
        index += 1
        return (index, argExprs + nexprs + [newAssNode], Name(newTemp))

    elif isinstance(node, Compare):
        if len(node.ops) == 1:
            index, exprs, exn = flatten(node.expr, index, varMap, strings)
            oper, rexp = node.ops[0]
            index, rexprs, rexn = flatten(rexp, index, varMap, strings)
            newTemp = "t{}".format(index)
            index += 1

            newCompNode = Compare(exn, [(oper, rexn)])
            newAssNode = Assign([AssName(newTemp, None)], newCompNode)
            return (index, exprs+rexprs+[newAssNode], Name(newTemp))
    
    elif isinstance(node, GetTag):
        index, exprs, n = flatten(node.arg, index, varMap, strings)
        newTemp = "t{}".format(index)
        index += 1
        newAssNode = Assign([AssName(newTemp, None)], GetTag(n))
        exprs.append(newAssNode)
        return (index, exprs, Name(newTemp))

    elif isinstance(node, Const):
        if not isinstance(node.value, int):
            raise Exception(node)
        return (index, [], node)

    elif isinstance(node, Name):
        if node.name not in varMap:
            raise Exception("{} referenced before assignment!".format(node.name))
        return (index, [], Name(varMap[node.name]))

    elif isinstance(node, FuncName):
        return (index, [], node)


    elif isinstance(node, ProjectTo):
        index, exprs, n = flatten(node.arg, index, varMap, strings)
        newTemp = "t{}".format(index)
        index += 1
        newAssNode = Assign([AssName(newTemp, None)], ProjectTo(node.typ, n))
        exprs.append(newAssNode)
        return (index, exprs, Name(newTemp))

    elif isinstance(node, InjectFrom):
        index, exprs, n = flatten(node.arg, index, varMap, strings)
        newTemp = "t{}".format(index)
        index += 1
        newAssNode = Assign([AssName(newTemp, None)], InjectFrom(node.typ, n))
        exprs.append(newAssNode)
        return (index, exprs, Name(newTemp))

    elif isinstance(node, Let):
        index, aboveExprs, above = flatten(node.sub, index, varMap, strings)

        scopedVmap = varMap.copy()
        newTemp = "t{}".format(index)
        index += 1
        newAssNode = Assign([AssName(newTemp, None)], above)
        aboveExprs.append(newAssNode)
        scopedVmap[node.var.name] = newTemp

        index, belowExprs, below = flatten(node.expr, index, scopedVmap, strings)

        return (index, aboveExprs + belowExprs, below)

    elif isinstance(node, Return):
        index, exprs, narg = flatten(node.value, index, varMap, strings)
        exprs.append(Return(narg))
        return (index, exprs, Name("don't use this"))
    
    else:
        print node
        raise Exception('unrecognized AST node {}'.format(node))

def removeSadness(assemblyAst):
    i = 0
    lenA = len(assemblyAst)
    while i<lenA:
        instr = assemblyAst[i]
        if isinstance(instr, Movl):
            #print "hi"+instr.source+"hi"+instr.dest+"hi"
            if isinstance(instr.source, Reg) and isinstance(instr.dest, Reg):
                if instr.source.name == instr.dest.name:
                    assemblyAst.remove(instr)
                    i-=1

        lenA = len(assemblyAst)
        i+=1
        if i>=lenA:
            break
    i = 0
    lenA = len(assemblyAst)
    while i<lenA-1:
        instr = assemblyAst[i]
        if isinstance(instr, Negl):
            instr2 = assemblyAst[i+1]
            if isinstance(instr2, Negl):
                if isinstance(instr.arg, Reg) and isinstance(instr2.arg, Reg):
                    if instr.arg.name == instr2.arg.name:
                        assemblyAst.remove(instr)
                        assemblyAst.remove(instr2)
                        i -= 2
        lenA = len(assemblyAst)
        i+=1
        if i>=lenA-1:
            break

    return 

def compilepy(sourceFile, targetFile, color_with_ilp, ilp_args, collect_accesses, greedyHint, hybridBalance):
    ast = compiler.parseFile(sourceFile)
    dast = declassify(ast, 0, None, gather_assignments(ast.node), set([]), {})
    u_env = {}
    uast, index = uniquify(dast, u_env, 0)
    explic = explicate(uast)
    #profiler(explic, "", {})
    freeVars = []
    compute_free_vars(explic.node, freeVars)
    freeVars = set(freeVars)
    tast = Module("", Stmt([Lambda([], [], 0, explic.node)]))
    hast = heapify(tast, freeVars)
    hast = Module("", hast.node.nodes[0].code)
    cast, flist, index = closure(hast, 0)

    strings = {}

    with open(targetFile, 'w+') as outfile:
        varMap = {n.name: n.name for n in flist}
        for f in reservedFunctions:
            varMap[f] = f
        for f in flist:
            index = 0
            fVarMap = dict(varMap)
            for i,n in enumerate(f.argnames):
                if not i:
                    fVarMap[n] = n
                else:
                    fVarMap[n] = "t{}".format(index)
                    f.argnames[i] = fVarMap[n]
                    index += 1
            count, mast = flatten(f.code, index, fVarMap, strings)
            f.code = mast
        count, mast = flatten(cast, 0, varMap, strings)

        for i,k in enumerate(strings):
            strings[k] = "str_{}".format(i)

        code = mast.node.nodes
        code.append(Return(Const(0)))
        mainFunc = Function(None, "main", [], [], 0, "", Stmt(code))
        flist.append(mainFunc)

        functionAssembly = []
        for f in flist:
            newAssemblyAst = []
            translate_ir(f, newAssemblyAst, strings)
            functionAssembly.append(newAssemblyAst)

        lastPruned = 0
        if len(strings):
            outfile.write(".data"+os.linesep)
            for k in strings:
                outfile.write("{}:".format(strings[k])+os.linesep)
                outfile.write("\t.string \"{}\"".format(k)+os.linesep)
            outfile.write(os.linesep)
        
        outfile.write(".global main"+os.linesep)
        accesses = 0
        for ast in functionAssembly:
            stackMap = {}
            lastStackLoc = 0
            passed = False
            while True:
                iGraph = {}
                colorMapping = {}
                liveSet = zombie(ast, set([]), iGraph, colorMapping)
                if color_with_ilp:
                    spills = {}
                    spillCost(ast, spills)
                    spillorder = [(k, spills[k]) for k in spills]
                    spillorder.sort(key=lambda x:x[1], reverse=True)

                    if greedyHint and not passed:
                        success, failedVar, colorMapping = color_fast(iGraph, colorMapping)
                        for v in colorMapping:
                            if colorMapping[v][1] == False:
                                p = random.random()
                                if p > hybridBalance:
                                    colorMapping[v] = (None, colorMapping[v][1])
                        passed = True

                    if not colorMapping:
                        success = True
                        failedVar = ""
                        coloring = {}
                    else:
                        success, failedVar, coloring = color_ilp(iGraph, colorMapping, spills, ilp_args)
                else:
                    success, failedVar, coloring = color_fast(iGraph, colorMapping)
                if (success):
                    assignRegisters(ast, coloring)
                    #removeSadness(ast)
                    assemblyAst, lastPruned = prune_if(ast, lastPruned)
                    if collect_accesses:
                        accesses += calculateAccesses(assemblyAst)
                    if lastStackLoc < 0:
                        assemblyAst = assemblyAst[:6] + [Subl(AConst(abs(lastStackLoc)), Reg("esp"))] + assemblyAst[6:-5] + [Addl(AConst(abs(lastStackLoc)), Reg("esp"))] + assemblyAst[-5:]
                    for i in assemblyAst:
                        if isinstance(i, Label):
                            outfile.write("{}".format(i)+os.linesep)
                        else:
                            outfile.write("\t{}".format(i)+os.linesep)
                    break
                else:
                    lastStackLoc -= 4
                    stackMap[failedVar] = lastStackLoc - 12
                    ast = adjustAst(ast, failedVar, stackMap)

        if collect_accesses:
            print "{},{}".format(sourceFile, accesses)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python to x86 Compiler (P1)")
    parser.add_argument('input_file', type=str, help="The Python file to compile")
    parser.add_argument('-color-ilp', action='store_true', default=False)
    parser.add_argument('-ilp-no-de-opt', action='store_false', default=True)
    parser.add_argument('-ilp-no-static-opt', action='store_false', default=True)
    parser.add_argument('-ilp-no-mem', action='store_false', default=True)
    parser.add_argument('-ilp-no-spill', action='store_false', default=True)
    parser.add_argument('-collect-accesses', action='store_true', default=False)
    parser.add_argument('-collect-constraints', action='store_true', default=False)
    parser.add_argument('-hybrid', action='store_true', default=False)
    parser.add_argument('-balance', type=int, default=50)
    args = parser.parse_args()   
    
    # TODO: handle the case where .py isn't the extension (let the parser handle syntax)
    compilepy(args.input_file, args.input_file.replace('.py', '.s'), args.color_ilp, (args.ilp_no_de_opt, args.ilp_no_static_opt, args.ilp_no_mem, args.collect_constraints, args.ilp_no_spill), args.collect_accesses, args.hybrid, args.balance/100.)
