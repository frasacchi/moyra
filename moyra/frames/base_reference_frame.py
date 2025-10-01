import sympy as sym
from sympy.abc import t
import sympy.physics.mechanics as me

class BaseReferenceFrame:

    def __init__(self,A=None,R=None):
        self._A = sym.eye(3) if A is None else sym.Matrix(A)
        self._R = sym.zeros(3,1) if R is None else sym.Matrix(R)

    A = property(lambda self: self._A)
    R = property(lambda self: self._R)    

    def transform_point(self,p):
        return self.A*sym.Matrix(list(p))+self.R

    def transform_global_point(self,p):
        return self.A.T*(sym.Matrix(list(p))-self.R)

    def transform_vector(self,v):
        return self.A*sym.Matrix(list(v))

    def transform_global_vector(self,v):
        return self.A.T*sym.Matrix(list(v))