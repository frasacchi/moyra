import sympy as sym

def MassMatrix(m,I_xx=0,I_yy=0,I_zz=0,I_xy=0,I_xz=0,I_yz=0):
    M = sym.diag(m,m,m,I_xx,I_yy,I_zz)
    M[4,3]=M[3,4]=I_xy
    M[5,3]=M[3,5]=I_xz
    M[5,4]=M[4,5]=I_yz
    return M