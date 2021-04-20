import sympy as sym
import numpy as np
from scipy.linalg import eig
from .helper_funcs import linearise_matrix

class NumericModel:

    def __init__(self,FwtParams,M,f,T,U,ExtForces = None):
        from sympy.abc import t
        """Initialise a Symbolic model of the form 
        $M\ddot{q}+f(\dot{q},q,t)-ExtForces(\dot{q},q,t) = 0$

        with the Symbolic Matricies M,f,and Extforces
        """
        p = FwtParams
  
        #generate lambda functions
        tup = p.GetTuple()
        
        # External Forces
        self.ExtForces = ExtForces.lambdify((tup,p.x,t)) if ExtForces is not None else lambda tup,x,t : 0
        # Mass Matrix Eqn
        self.M_func = sym.lambdify((tup,p.x),M,"numpy")
        #func eqn
        self.f_func = sym.lambdify((tup,p.x),f,"numpy")
        # potential energy function
        self.u_eqn = sym.lambdify((tup,p.x),U,"numpy")
        # kinetic energy function
        self.t_eqn = sym.lambdify((tup,p.x),T,"numpy")

    @classmethod
    def from_SymbolicModel(cls,p,sm):
        return cls(p,sm.M,sm.f,sm.T,sm.U,sm.ExtForces)


    def deriv(self,t,x,tup):
        try:
            external = self.ExtForces(tup,x,t)
            accels = np.linalg.inv(self.M_func(tup,x))@(-self.f_func(tup,x)+external)
        except ZeroDivisionError:
            accels = np.linalg.inv(self.M_func(tup,x))@(-self.f_func(tup,x))        

        state_vc = []
        for i in range(0,int(len(x)/2)):
            state_vc.append(x[(i)*2+1])
            state_vc.append(accels[i,0])
        return tuple(state_vc)

    #calculate the total energy in the system
    def ke(self,t,x,tup):
        return self.t_eqn(tup,x)

    def pe(self,t,x,tup):
        return self.u_eqn(tup,x)

    def energy(self,t,x,tup):
        return self.ke(t,x,tup) + self.pe(t,x,tup)