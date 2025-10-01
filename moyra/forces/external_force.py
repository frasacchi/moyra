import sympy as sym
import sympy.physics.mechanics as me
from inspect import getsource
from ..helper_funcs import linearise_matrix


class ExternalForce:

    @classmethod
    def zero_force(cls,dofs):
        Q = sym.Matrix([0]*dofs)
        return cls(Q)

    @classmethod
    def body_force(cls,q,frame,Fx=0,Fy=0,Fz=0,Mx=0,My=0,Mz=0,simplify=True):
        BodyJacobian = frame.BodyJacobian(q)
        BodyJacobian = sym.simplify(cls._trigsimp(BodyJacobian)) if simplify else BodyJacobian
        wrench = sym.Matrix([Fx,Fy,Fz,Mx,My,Mz])
        return cls(BodyJacobian.T*wrench) 

    @staticmethod
    def _trigsimp(expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr)))) 

    def __init__(self,Q = None):
        self._Q = Q

    def __mul__(self,other):
        return ExternalForce(self._Q*other)

    def __add__(self,other):
        if isinstance(other, ExternalForce):
            return ExternalForce(self._Q+other._Q)
        else:
            return ExternalForce(self._Q+other)

    def Q(self):
        return self._Q

    def subs(self,*args):
        return ExternalForce(self._Q.subs(*args))

    def msubs(self,*args):
        return ExternalForce(me.msubs(self._Q,*args))

    def cancel(self):
        return ExternalForce(sym.cancel(self._Q))
    
    def simplify(self):
        return ExternalForce(sym.simplify(self._Q))

    def expand(self):
        return ExternalForce(sym.expand(self._Q))  

    def integrate(self,*args):
        return ExternalForce(self._Q.integrate(*args))

    def linearise(self,x,x_f):
        return ExternalForce(linearise_matrix(self.Q(),x,x_f))

    def lambdify(self,params):
        if self._Q is None:
            return None
        return sym.lambdify(params,self._Q ,"numpy")




    





