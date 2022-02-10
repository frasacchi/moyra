import sympy as sym
from .base_element import BaseElement
from .mass_matrix import MassMatrix
from sympy.physics.mechanics import dot

class RigidElement(BaseElement):
    def __init__(self,Transform, M, gravityPotential=False, com_pos=[0,0,0], simplify=True):
        self._gravityPotential = gravityPotential
        self.Transform = Transform
        self.M_e = M
        self.com_pos = com_pos
        self.simplify = simplify

    @classmethod
    def point_mass(cls, Transform,m,gravityPotential=False):
        return cls(Transform,MassMatrix(m),gravityPotential)    
    
    def calc_ke(self,p):
        M = self.M(p)   

        # calculate the K.E
        T = sym.Rational(1,2)*p.qd.T*M*p.qd
        return self._trigsimp(T[0]) if self.simplify else T[0]

    def M(self,p):
        # create the jacobian for the mass
        T = self.Transform.Translate(*self.com_pos)
        Js = T.ManipJacobian(p.q)
        Jb = T.InvAdjoint()*Js
        Jb = self._trigsimp(Jb) if self.simplify else Jb
        #get M in world frame
        #calculate the mass Matrix
        return Jb.T*self.M_e*Jb

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))


    def calc_pe(self,p):
        if self._gravityPotential:
            point = self.Transform.Transform_point(self.com_pos)
            h = -(point.T*p.g_v)[0]
            return h*self.M_e[0,0]*p.g
        else:
            return 0
    
    def calc_rdf(self,p):
        return 0