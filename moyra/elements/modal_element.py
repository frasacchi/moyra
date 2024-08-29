from functools import cache
from moyra.elements.mass_matrix import MassMatrix
from moyra.frames.homogenous_frame import HomogenousFrame
from moyra.frames.reference_frame import ReferenceFrame
from moyra.helper_funcs import Wedge
import sympy as sym
from sympy.abc import t
from .base_element import BaseElement
from sympy.physics.mechanics import msubs,dot
from collections.abc import Iterable

class ModalElement(BaseElement):
    def __init__(self,q,frame,q_f,M_trace,K_trace,simplify=True, name="default"):

        self._frame = frame
        self._q_f = sym.Matrix(q_f) if isinstance(q_f,Iterable) else sym.Matrix([q_f])
        self._M_trace = M_trace
        self._K_trace = K_trace

        if isinstance(frame,HomogenousFrame):
            q = q
        else:
            raise ValueError('Invlaid Transform type')

        super(ModalElement, self).__init__(q,name)
        self._simplify = simplify
    
    q_f = property(lambda self: self._q_f)
    frame = property(lambda self: self._frame)
    M_trace = property(lambda self: self._M_trace)
    K_trace = property(lambda self: self._K_trace)
    simplify = property(lambda self: self._simplify)

    @property
    @cache
    def ke(self):
        # calculate the K.E
        T = sym.Rational(1,2)*self.qd.T*self.M*self.qd
        return T[0]
    
    @property
    @cache
    def pe(self):
        return self.elastic_pe + self.grav_pe

    @property
    @cache
    def elastic_pe(self):
        U = sym.Rational(1,2)*self.q_f.T*sym.diag(*self.K_trace)*self.q_f
        return U[0]
    
    @property
    @cache
    def grav_pe(self):
        return sym.Integer(0)

    @property
    def rdf(self):
        return sym.Integer(0)

    @property
    @cache
    def M(self):
        M = sym.zeros(len(self.q))
        idx = [i for i,val in enumerate(self.q_f) if val in self.q]
        for i in idx:
            M[i,i] = self.M_trace[i]
        return M

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))



            
