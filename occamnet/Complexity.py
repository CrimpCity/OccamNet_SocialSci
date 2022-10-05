import numpy as np
from sympy import Rational
from sympy.parsing.sympy_parser import parse_expr
from sympy import preorder_traversal, count_ops

def bestApproximation(x,imax):
    # The input is a numpy parameter vector p.
    # The output is an integer specifying which parameter to change,
    # and a float specifying the new value.
    def float2contfrac(x,nmax):
        x = float(x)
        c = [np.floor(x)];
        y = x - np.floor(x)
        k = 0
        while np.abs(y)!=0 and k<nmax:
            y = 1 / float(y)
            i = np.floor(y)
            c.append(i)
            y = y - i
            k = k + 1
        return c
    
    def contfrac2frac(seq):
        ''' Convert the simple continued fraction in `seq`
            into a fraction, num / den
            '''
        num, den = 1, 0
        for u in reversed(seq):
            num, den = den + num*u, num
        return num, den
    
    def contFracRationalApproximations(c):
        return np.array(list(contfrac2frac(c[:i+1]) for i in range(len(c))))
    
    def contFracApproximations(c):
        q = contFracRationalApproximations(c)
        return q[:,0] / float(q[:,1])
    
    def truncateContFrac(q,imax):
        k = 0
        while k < len(q) and np.maximum(np.abs(q[k,0]), q[k,1]) <= imax:
            k = k + 1
        return q[:k]
    
    def pval(p):
        p = p.astype(float)
        return 1 - np.exp(-p ** 0.87 / 0.36)
    
    xsign = np.sign(x)
    q = truncateContFrac(contFracRationalApproximations(float2contfrac(abs(x),20)),imax)
    
    if len(q) > 0:
        p = np.abs(q[:,0] / q[:,1] - abs(x)).astype(float) * (1 + np.abs(q[:,0])) * q[:,1]
        p = pval(p)
        i = np.argmin(p)
        return (xsign * q[i,0] / float(q[i,1]), xsign* q[i,0], q[i,1], p[i])
    else:
        return (None, 0, 0, 1)

def get_number_DL_snapped(n):
    epsilon = 1e-10
    n = float(n)
    if np.isnan(n):
        return 1000000
    elif np.abs(n - int(n)) < epsilon:
        return np.log2(1 + abs(int(n)))
    elif np.abs(n - bestApproximation(n,10000)[0]) < epsilon:
        _, numerator, denominator, _ = bestApproximation(n, 10000)
        return np.log2((1 + abs(numerator)) * abs(denominator))
    elif np.abs(n - np.pi) < epsilon:
        return np.log2(1+3)
    else:
        PrecisionFloorLoss = 1e-14
        return np.log2(1 + (float(n) / PrecisionFloorLoss) ** 2) / 2

def get_expr_complexity(expr):
    expr = parse_expr(expr)
    compl = 0

    is_atomic_number = lambda expr: expr.is_Atom and expr.is_number
    numbers_expr = [subexpression for subexpression in preorder_traversal(expr) if is_atomic_number(subexpression)]

    for j in numbers_expr:
        try:
            compl = compl + get_number_DL_snapped(float(j))
        except:
            compl = compl + 1000000

    n_variables = len(expr.free_symbols)
    n_operations = len(count_ops(expr,visual=True).free_symbols)

    if n_operations!=0 or n_variables!=0:
        compl = compl + (n_variables+n_operations)*np.log2((n_variables+n_operations))

    return compl