import sympy as sym
import sympy.physics.mechanics as me
from inspect import getsource
from ..helper_funcs import LineariseMatrix


class ExternalForce:

    def __init__(self,Q = None):
        self._Q = Q

    def __mul__(self,other):
        return ExternalForce(self._Q*other)

    def Q(self):
        return self._Q

    def subs(self,*args):
        return ExternalForce(self._Q.subs(*args))

    def msubs(self,*args):
        return ExternalForce(me.msubs(self._Q,*args))

    def cancel(self):
        return ExternalForce(sym.cancel(self._Q))  

    def expand(self):
        return ExternalForce(sym.expand(self._Q))  

    def integrate(self,*args):
        return ExternalForce(self._Q.integrate(*args))

    def linearise(self,x,x_f):
        return ExternalForce(LineariseMatrix(self.Q(),x,x_f))

    def lambdify(self,params):
        if self._Q is None:
            return None
        return sym.lambdify(params,self._Q ,"numpy")




    





