"""Compare MATLAB and C to solve EMD distance problem"""
import numpy as np
import scipy.io as sio
# from openopt import LP
from scipy.spatial.distance import cdist
from emd_dst import dist_for_emd
from emd import emd
DIM = 31
NP = 10
NQ = 10
P, Q, w1, w2, lP, lQ, fw1, fw2 = None, None, None, None, None, None, None, None




def profile(f):
    return f


def initial_solution(weights1, weights2, max_cost):
    """Find an initial solution using russel method from the Yossi Rubner
    implemention."""
    from subprocess import check_output
    max_weights = max(np.max(weights1), np.max(weights2))
    inarg = '{} {}\n{} {}\n'.format(weights1.size, weights2.size, max_cost,
                                    max_weights)
    inarg += ' '.join([str(float(_)) for _ in weights1])
    inarg += '\n' + ' '.join([str(float(_)) for _ in weights1])
    with open('in', 'w') as args:
        args.write(inarg)
    res = check_output('./a.out < in'.format(inarg), shell=True)
    return [float(_) for _ in res.split()]


@profile
def write_matlab_problem(idx, mlab=None):
    """Write in a file the variables describing the `idx`th problem to be
    solved"""
    costs = cdist(P, Q)
    vcost = costs.ravel('F')
    i_w = np.kron(np.ones((1, costs.shape[1])), np.eye(costs.shape[0]))
    j_w = np.kron(np.eye(costs.shape[1]), np.ones((1, costs.shape[0])))
    Aeq = np.vstack([i_w, j_w])
    Aeq = np.insert(Aeq, Aeq.shape[0], 1, axis=0)
    beq = np.vstack([w1.reshape(w1.size, 1), w2.reshape(w2.size, 1), [[1]]])
    A = np.vstack([i_w, j_w])
    A = np.insert(A, A.shape[0], -1, axis=0)
    b = np.vstack([w1.reshape(w1.size, 1), w2.reshape(w2.size, 1), [[-.97]]])
    lb = len(vcost)*[0, ]
    ub = len(vcost)*[np.inf, ]
    # f = initial_solution(w1, w2, np.max(costs))
    # p = LP(d, lb=lb, ub=ub,
    #        A=A, b=b,
    #        # Aeq=Aeq, beq=beq,
    #        )
    # return (dir(p.minimize))
    sio.savemat('{}/{}_{}'.format('/tmp/mats', 'lpin', idx),
                {'f': vcost, 'A':A, 'b': b}, do_compression=True)
    # mlab.run_func('mlinprog.m')
    # res = sio.loadmat('lpout')
    # r2 = p.minimize('glpk', iprint=-1)
    # print(r2.ff)
    # return r2.ff


MATLAB_CMD = "clear all;mlinprog({});"
@profile
def collect_matlab_output(nb_input, wipeout=False):
    """Call MATLAB to solve the `nb_input` first problems and return the list
    of EMD cost"""
    from subprocess import check_call
    real_cmd = MATLAB_CMD.format(nb_input)
    check_call('matlabd "{}"'.format(real_cmd), shell=True)
    costs = []
    for idx in range(nb_input):
        filename = '{}/{}_{}.mat'.format('/tmp/mats', 'lpout', idx)
        try:
            dst = sio.loadmat(filename)['dst']
            if wipeout:
                os.remove(filename)
                os.remove(filename.replace('lpout', 'lpin'))
        except IOError:
            dst = 1e15
        costs.append(float(dst))
    return costs


@profile
def solve_by_emd():
    """Solve the problem where the whole mass must be moved."""
    ltheta = DIM*[1, ]
    return emd((lP, fw1), (lQ, fw2),
               lambda a, b: float(dist_for_emd(a, b, ltheta)))


if __name__ == '__main__':
    nb_test = 2
    for _ in range(nb_test):
        P = np.random.rand(NP, DIM)
        Q = np.random.rand(NQ, DIM)
        w1 = np.random.rand(P.shape[0])
        w2 = np.random.rand(Q.shape[0])
        w1 /= np.sum(w1)
        w2 /= np.sum(w2)
        lP = P.tolist()
        lQ = Q.tolist()
        fw1 = list(map(float, w1))
        fw2 = list(map(float, w2))
        # write_matlab_problem(_)
        print(solve_by_emd())
    import sys
    # print(collect_matlab_output(nb_test))
    sys.exit()
