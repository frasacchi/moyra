from . import ExternalForce
import sympy as sym

class ZeroForce(ExternalForce):

    def __init__(self,dofs):
        _Q = sym.Matrix([0]*dofs)
        super().__init__(_Q)

        
