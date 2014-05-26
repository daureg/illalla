cimport cython
@cython.boundscheck(False)
def dist_for_emd(a1, a2):
    s=0
    for i in range(len(a1)):
        s += (a1[i] - a2[i])*(a1[i] - a2[i])
    return s
