import sympy as sym
from .base_element import BaseElement
from .mass_matrix import MassMatrix

class RigidElement(BaseElement):
    def __init__(self,Transform,M,gravityPotential=False,com_pos = [0,0,0]):
        self._gravityPotential = gravityPotential
        self.Transform = Transform
        self.M_e = M
        self.com_pos = com_pos

    @classmethod
    def point_mass(cls, Transform,m,gravityPotential=False):
        return cls(Transform,MassMatrix(m),gravityPotential)    
    
    def calc_ke(self,p):
        M = self.M(p)   

        # calculate the K.E
        T = sym.Rational(1,2)*p.qd.T*M*p.qd
        return self._trigsimp(T[0])

    def M(self,p):
        # create the jacobian for the mass
        Js = self.Transform.ManipJacobian(p.q)
        Jb = self.Transform.InvAdjoint()*Js
        Jb = self._trigsimp(Jb)
        #get M in world frame
        #calculate the mass Matrix
        return Jb.T*self.M_e*Jb

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))


    def calc_pe(self,p):
        if self._gravityPotential:
            #return 0
            point = self.Transform.Transform_point(self.com_pos)
            return point[2]*self.M_e[0,0]*p.g
        else:
            return 0
    
    def calc_rdf(self,p):
        return 0