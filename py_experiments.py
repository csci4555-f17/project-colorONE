from assembly import *

def calculateAccesses(ast):
    a = 0
    for instr in ast:
        if isinstance(instr, CallInd):
            if isinstance(instr.target, Mem):
                a += 1
        elif isinstance(instr, Xorl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Andl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Sarl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Orl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Shl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Cmpl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Pushl):
            if isinstance(instr.arg, Mem):
                a += 1
        elif isinstance(instr, Movl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Movzbl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Subl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Addl):
            if isinstance(instr.source, Mem):
                a += 1
            if isinstance(instr.dest, Mem):
                a += 1
        elif isinstance(instr, Negl):
            if isinstance(instr.arg, Mem):
                a += 1

    return a
