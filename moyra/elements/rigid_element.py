from functools import cache
from moyra.frames.homogenous_frame import HomogenousFrame
from moyra.frames.reference_frame import ReferenceFrame
from moyra.helper_funcs import Wedge,Vee
import sympy as sym
from sympy.abc import t
from .base_element import BaseElement
from .mass_matrix import MassMatrix

class RigidElement(BaseElement):

    @classmethod
    def point_mass(cls,q,frame,m,grav_vec=sym.Matrix([0]*3),simplify=True):
        return cls(q,frame,MassMatrix(m),grav_vec,simplify=True)  

    def __init__(self, q, frame, M, grav_vec=sym.Matrix([0]*3), com_pos=[0,0,0], simplify=True, name = "default", alt_method=False):
        self._grav_vec = grav_vec
        self._frame = frame
        if M.shape == (6,6):
            self._M_e = M
        else:
            raise ValueError('Invlaid Mass matrix')
        # self._M_e = M
        self._com_pos = com_pos
        self._simplify = simplify
        self._alt_method = alt_method
        if isinstance(frame,HomogenousFrame):
            q = q
            self._idx =  list(range(len(q)))
        elif isinstance(frame,ReferenceFrame):
            q = sym.Matrix([*frame.R,*frame.theta])
            self._idx = [i for i,v in enumerate(q) if (v.args[0] == t if isinstance(v,sym.Function) else False)]
            q = sym.Matrix(q[self._idx,:])
        else:
            raise ValueError('Invlaid Transform type')

        super(RigidElement, self).__init__(q,name)
    
    M_e = property(lambda self: self._M_e)
    com_pos = property(lambda self: self._com_pos)
    frame = property(lambda self: self._frame)
    grav_vec = property(lambda self: self._grav_vec)
    simplify = property(lambda self: self._simplify)
    rdf = property(lambda self: 0)
    alt_method = property(lambda self: self._alt_method)

    @property
    @cache
    def ke(self):
        if self.M_e[0,0]==0:
            return 0
        if isinstance(self.frame,HomogenousFrame):
            Rf = self.frame.Translate(*self.com_pos)
        if isinstance(self.frame,ReferenceFrame):
            Rf = self.frame
        W = sym.Matrix([*Rf.R.diff(t),*Vee(Rf.A.T*Rf.A.diff(t))])
        T =  (sym.Rational(1,2) * W.T * self.M_e * W)[0]
        return T.expand() if self.simplify else T

    @property
    @cache
    def M(self):
        #get M in world frame
        #calculate the mass Matrix
        if self.alt_method:
            M = sym.Matrix([self.ke]).jacobian(self.q.diff(t)).diff(t).jacobian(self.q.diff(t,2))
        else:
            if isinstance(self.frame,HomogenousFrame):
                T = self.frame if sum(self.com_pos)==0 else self.frame.Translate(*self.com_pos)
                Js = T.ManipJacobian(self.q)
                Js = self._trigsimp(Js) if self.simplify else Js
                IA = self._trigsimp(T.InvAdjoint()) if self.simplify else T.InvAdjoint()
                Jb = self._trigsimp(IA*Js) if self.simplify else IA*Js
                M = Jb.T*self.M_e*Jb
            if isinstance(self.frame,ReferenceFrame):
                M_rr = sym.eye(3)*self.M_e[0,0]
                r = Wedge(self.com_pos)
                M_rtheta = -self.M_e[0,0]*self.frame.A*r*self.frame.Gb
                M_thetatheta = self.frame.Gb.T*(self.M_e[3:,3:] + self.M_e[0,0]*r.T*r)*self.frame.Gb
                M = sym.BlockMatrix([[M_rr,M_rtheta],[M_rtheta.T,M_thetatheta]]).as_explicit()   
            M = M[self._idx,self._idx]    
        return self._trigsimp(M) if self.simplify else M


    @property
    @cache
    def pe(self):
        point = self.frame.transform_point(self.com_pos)
        h = -(point.T*self.grav_vec)[0]
        return h*self.M_e[0,0] if h != 0 else 0
    
    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))