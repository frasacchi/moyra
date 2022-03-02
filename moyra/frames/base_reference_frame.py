import sympy as sym
from sympy.abc import t
import sympy.physics.mechanics as me

class BaseReferenceFrame:

    def __init__(self,A=None,R=None):
        self.A = sym.eye(3) if A is None else sym.Matrix(A)
        self.R = sym.zeros(3,1) if R is None else sym.Matrix(R)

    def Transform_point(self,p):
        return self.A*sym.Matrix(list(p))+self.R

    def Transform_vector(self,v):
        return self.A*sym.Matrix(list(v))