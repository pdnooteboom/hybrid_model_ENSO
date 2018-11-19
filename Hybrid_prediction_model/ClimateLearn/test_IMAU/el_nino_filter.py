__author__ = 'ruggero'

import matplotlib.pyplot as plt


def filter(t,results,width = 0.4,spacing = 0.05):
    """
    Filtering function for classification methods.
    :param t: A list with time values
    :param results: A list of predicted classified results
    :param width: The width of the window for minimum positive classification
    :param spacing: The minimum spacing to merge small time windows
    :return: returns a list with filtered results
    """
    init = []
    fin = []
    flag = 0
    for i in range(0,len(t)):
        if (flag == 0)and(results[i] == 1):
            flag = 1
            init.append(i)
        if (flag == 1)and(results[i] == 0):
            flag = 0
            fin.append(i)
    if len(init) != len(fin):
        fin.append(results[len(t)-1])
    assert len(init) == len(fin)


    res_filt = [0]*len(results)
    for i in range(0,len(init)):
        if t[fin[i]]-t[init[i]] >= width:
            for j in range(init[i],fin[i]+1):
                    res_filt[j] = 1
        elif (i < len(init)-1)and(t[init[i+1]]-t[fin[i]] < spacing):
            t[init[i+1]] = t[init[i]]

    return res_filt


def full_filter(t,nino,width = 0.4):
    assert len(t) == len(nino)
    init = []
    fin = []
    flag = 0
    for i in range(0,len(t)):
        if (flag == 0)and(nino[i] == 1):
            flag = 1
            init.append(i)
        if (flag == 1)and(nino[i] == 0):
            flag = 0
            fin.append(i-1)
    if len(init) != len(fin):
        fin.append(len(t)-1)
    assert len(init) == len(fin)

    step = t[1]-t[0]
    inter = []
    dist = []
    for i in range(0,len(init)-1):
        inter.append(step*(fin[i]-init[i]))
        dist.append(step*(init[i+1]-fin[i]))
    inter.append(step*(fin[-1]-init[-1]))

    res_filt = [0]*len(t)
    j = 0
    final = 0
    while True:
        k = j+1
        while True:
            print k
            if dist[k-1] > width:
                break
            else:
                k=k+1


            if k == len(inter)-1:
                final = 1
                break
        if final == 1:
            if step*(fin[-1]- init[j]) > width:
                for i in range(init[j],fin[-1]+1):
                    res_filt[i] = 1
            break

        if step*(fin[k-1]- init[j]) > width:
            for i in range(init[j],fin[k-1]+1):
                res_filt[i] = 1
        j = k
    return res_filt
