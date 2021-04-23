import sympy as sym
import numpy as np
from scipy.linalg import eig
from .helper_funcs import linearise_matrix

class NumericModel:

    def __init__(self,p,M,f,T,U,ExtForces = None):
        from sympy.abc import t
        """Initialise a Symbolic model of the form 
        $M\ddot{q}+f(\dot{q},q,t)-ExtForces(\dot{q},q,t) = 0$

        with the Symbolic Matricies M,f,and Extforces
        """
  
        #generate lambda functions
        tup = p.GetTuple()
        
        # External Forces
        self.ExtForces = ExtForces.lambdify((p.x,tup,t)) if ExtForces is not None else lambda x,tup,t : 0
        # Mass Matrix Eqn
        self.M = sym.lambdify((p.x,tup),M,"numpy")
        #func eqn
        self.f = sym.lambdify((p.x,tup),f,"numpy")
        # potential energy function
        self.pe = sym.lambdify((p.x,tup),U,"numpy")
        # kinetic energy function
        self.ke = sym.lambdify((p.x,tup),T,"numpy")

    @classmethod
    def from_SymbolicModel(cls,p,sm):
        return cls(p,sm.M,sm.f,sm.T,sm.U,sm.ExtForces)


    def deriv(self,x,tup,t):
        try:
            external = self.ExtForces(x,tup,t)
            accels = np.linalg.inv(self.M(x,tup))@(-self.f(x,tup)+external)
        except ZeroDivisionError:
            accels = np.linalg.inv(self.M(x,tup))@(-self.f(x,tup))        

        state_vc = []
        for i in range(0,int(len(x)/2)):
            state_vc.append(x[(i)*2+1])
            state_vc.append(accels[i,0])
        return tuple(state_vc)

    def energy(self,x,tup,t):
        return self.ke(x,tup) + self.pe(x,tup)