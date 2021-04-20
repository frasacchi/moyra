import sympy as sym
import numpy as np
from . import ExternalForce

class CompositeForce(ExternalForce):

    def __init__(self,forces):
        self.forces = forces

    def Q(self):
        filt = filter(None,(f.Q() for f in self.forces))
        return sum(filt,next(filt))

    def subs(self,*args):
        return CompositeForce([force.subs(*args) for force in self.forces])

    def msubs(self,*args):
        return CompositeForce([force.msubs(*args) for force in self.forces])

    def cancel(self):
        return CompositeForce([force.cancel() for force in self.forces])

    def expand(self):
        return CompositeForce([force.expand() for force in self.forces])

    def linearise(self,x,x_f):
        return CompositeForce([force.linearise(x,x_f) for force in self.forces])

    def lambdify(self,params):
        force_funcs = list(filter(None,[force.lambdify(params) for force in self.forces]))
        return CallableCompositeForce(force_funcs)

class CallableCompositeForce:
    def __init__(self,force_funcs):
        self.force_funcs = force_funcs
    
    def __call__(self,*params):
        return sum((force_func(*params) for force_func in self.force_funcs))