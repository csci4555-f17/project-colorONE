from pulp import *



def color_ilp(iGraph, colorMapping, args):
    DOUBLE_EDGE_OPT = args[0]
    STATIC_COLOR_OPT = args[1]
    MEM_VAR = args[2]
    
    vertexMap = {vertex: [] for vertex in iGraph}
    memoryMap = {}
    prob = LpProblem("problem", LpMinimize)

    tccount = 0
    varcount = 0
    for vertex in colorMapping:
        varcount += 1

    # create variables for each possible color for each vertex
    for vertex in iGraph:
        for color in range(1, 7):
            vertexMap[vertex].append(LpVariable("{}_{}".format(vertex, color), 0, 1, cat='Integer'))
        if MEM_VAR and not colorMapping[vertex][1]:
            memoryMap[vertex] = LpVariable("{}_mem".format(vertex), 0, 1, cat='Integer')
        
    # enforce that only one color may be chosen for each vertex
    for vertex in vertexMap:
        if MEM_VAR and not colorMapping[vertex][1]:
            prob += sum(vertexMap[vertex]) + memoryMap[vertex] == 1
        else:
            prob += sum(vertexMap[vertex])  == 1
        tccount += 1

    # enforce pre-colorings
    for vertex in colorMapping:
        if colorMapping[vertex][0] is not None:
            prob += vertexMap[vertex][colorMapping[vertex][0]-1] == 1
            tccount += 1

    # enforce that each pair of adjacent vertices can't be the same color
    added = {}
    for vertex in iGraph:
        for adjVertex in iGraph[vertex]:
            if DOUBLE_EDGE_OPT:
                if vertex in added:
                    if adjVertex in added[vertex]:
                        continue
                elif adjVertex in added:
                    if vertex in added[adjVertex]:
                        continue
                else:
                    added[vertex] = {}

                added[vertex][adjVertex] = 1

            possV1 = vertexMap[vertex]
            possV2 = vertexMap[adjVertex]

            if STATIC_COLOR_OPT:
                if colorMapping[vertex][0] is not None:
                    if colorMapping[adjVertex][0] is not None:
                        continue
                    color = colorMapping[vertex][0] - 1
                    prob += possV2[color] == 0
                    tccount += 1
                    continue
                elif colorMapping[adjVertex][0] is not None:
                    color = colorMapping[adjVertex][0] - 1
                    prob += possV1[color] == 0
                    tccount += 1
                    continue

            for i in range(len(possV1)):
                prob += possV1[i] + possV2[i] <= 1
                tccount += 1

    print "variables: {}, constraints: {}".format(varcount, tccount)

    # objective
    if MEM_VAR:
        memVars = [memoryMap[vertex] for vertex in memoryMap]
        prob += sum(memVars)
    else:
        totalVarList = [v for k in vertexMap for v in vertexMap[k]]
        prob += totalVarList[0]
    
    #status = prob.solve(GLPK(msg=0))
    status = prob.solve(GLPK(msg=0, options=["--binarize"]))
    if status == 1:
        for vertex in vertexMap:
            found = False
            for i,c in enumerate(vertexMap[vertex]):
                if value(c):
                    colorMapping[vertex] = ((i+1), False)
                    found = True
                    break
            if not found:
                return False, vertex, colorMapping
        return True, None, colorMapping
    else:
        print LpStatus[status]
        exit(0)
    

