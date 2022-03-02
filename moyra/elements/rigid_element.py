from moyra.frames.homogenous_frame import HomogenousFrame
from moyra.frames.reference_frame import ReferenceFrame
from moyra.helper_funcs import Wedge
import sympy as sym
from .base_element import BaseElement
from .mass_matrix import MassMatrix

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
    
    def calc_ke(self,p,M=None):
        M = M if M is not None else self.M(p)   
        # calculate the K.E
        T = sym.Rational(1,2)*p.qd.T*M*p.qd
        return self._trigsimp(T[0]) if self.simplify else T[0]

    def M(self,p):
        #get M in world frame
        #calculate the mass Matrix
        if isinstance(self.Transform,HomogenousFrame):
            T = self.Transform if sum(self.com_pos)==0 else self.Transform.Translate(*self.com_pos)
            Js = T.ManipJacobian(p.q)
            Js = self._trigsimp(Js) if self.simplify else Js
            Jb = T.InvAdjoint()*Js
            Jb = self._trigsimp(Jb) if self.simplify else Jb
            M = Jb.T*self.M_e*Jb
        if isinstance(self.Transform,ReferenceFrame):
            M_rr = sym.eye(3)*self.M_e[0,0]
            r = Wedge(self.com_pos)
            M_rtheta = -self.M_e[0,0]*self.Transform.A*r*self.Transform.Gb
            M_thetatheta = self.Transform.Gb.T*(self.M_e[3:,3:] + self.M_e[0,0]*r.T*r)*self.Transform.Gb
            M = sym.BlockMatrix([[M_rr,M_rtheta],[M_rtheta.T,M_thetatheta]]).as_explicit()       
        return M

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