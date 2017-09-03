from scipy.sparse import *
from scipy.sparse.linalg import *
from scipy.optimize import *
from numpy import *

def get_bit(z, i):
    """
    gets the i'th bit of the integer z (0 labels least significant bit)
    """
    return (z >> i) & 0x1

def validate(n, g, z):
    """
    checks if z (an integer) represents a valid vertex cover for graph adjacency
    matrix g, with n vertices
    """
    for i in range(0, n):
        for j in range(0, i):
            if g[i, j] == 1 and get_bit(z, i) == 0 and get_bit(z, j) == 0:
                return False
    return True

def c(n, z):
    """
    the objective function to maximise: counts the number of 0 bits in an integer,
    corresponding to maximising the number of vertices NOT in the vertex cover
    """
    s = 0
    for i in range(0, n):
        s += get_bit(z, i)
    return n - s

def c_op(n):
    """
    a 2^n by 2^n diagonal matrix with c(n, z) entries on the diagonal
    """
    return diags([c(n, z) for z in range(0, 2**n)])

def uc(cop, gamma, v):
    """
    performs the matrix operation exp(-i gamma cop).v
    """
    return expm_multiply(-1.0j * gamma * cop, v)

def istate(n):
    """
    constructs the initial state |1....1>
    """
    N = 2**n
    v = lil_matrix((N, 1), dtype=dtype(cfloat))
    v[N-1, 0] = 1
    return csc_matrix(v)

def b_op(n, g):
    """
    a 2^n by 2^n sparse matrix which allows a CTQW between
    quantum states representing valid vertex covers on the graph g
    (which has exactly n vertices)
    """
    N = 2**n
    b = lil_matrix((N, N), dtype=dtype(int))
    for x1 in range(0, N):
        if not validate(n, g, x1):
            continue
        for j in range(0, n):
            bit = 0x1 << j
            x2 = x1 ^ bit
            if validate(n, g, x2):
                b[x1, x2] = 1
    return csc_matrix(b)

def ub(bop, b, v):
    """
    performs the matrix operation exp(-i b bop).v
    """
    return expm_multiply(-1.0j * b * bop, v)

def evolve(n, p, bs, gs, bop, cop):
    """
    constructs the final state |b, gamma> from the initial state
    by evolving a n-qubit quantum state with p applications of bop
    and (p-1) applications of cop, as per QAOA
    """
    v = istate(n)
    for i in range(0, p - 1):
        v = ub(bop, bs[i], v)
        v = uc(cop, gs[i], v)
    v = ub(bop, bs[p-1], v)
    return v

def probs(v):
    """
    returns a 2^n by 1 vector of probabilities associated with each
    computational basis state for a quantum state v
    """
    return real(v.multiply(v.conj()))

def expectation(n, p, bs, gs, bop, cop):
    """
    finds the expectation value of the final QAOA state with respect
    to the C operator
    """
    array = probs(evolve(n, p, bs, gs, bop, cop))
    return sum([c(n, z) * array[z] for z in range(0, 2**n)])[0, 0]

def best(n, g):
    """
    brute-forces a solution to the vertex cover problem and returns the
    number of vertices in the optimal solution
    """
    return max([c(n, z) if validate(n, g, z) else 0 for z in range(0, 2**n)])

def _f_expectation_for_optimizer(n, p, bop, cop):
    """
    essentially the objective function that is passed to the optimizer
    """
    def f(params):
        bs = params[0:p]
        gs = params[p:]
        return -expectation(n, p, bs, gs, bop, cop)
    return f

def qaoa(n, g, p, bop, cop):
    """
    performs Nelder-Mead optimization to find the best set of parameters for the QAOA algorithm
    """
    return minimize(_f_expectation_for_optimizer(n, p, bop, cop), [0]*(2*p-1), method='Nelder-Mead')

def qaoa_quality(n, g, p, bop, cop):
    """
    returns a quality from [0-1] of the result of the QAOA algorithm as compared to the optimal solution
    """
    result = qaoa(n, g, p, bop, cop)
    bs = result.x[0:p]
    gs = result.x[p:]
    array = probs(evolve(n, p, bs, gs, bop, cop))
    ttotal = 0
    ctotal = 0
    for z in range(0, 2**n):
        curr = c(n, z)
        if validate(n, g, z) and curr >= -result.fun:
            ctotal += array[z, 0]
            ttotal += curr * array[z, 0]
    return ttotal / (ctotal * best(n, g))

if __name__ == "__main__":
    # quick test
    g = csc_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0]])
    n = g.shape[0]
    p = 2
    print qaoa_quality(n, g, p, b_op(n, g), c_op(n))
