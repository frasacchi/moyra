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

class FlexiElement(BaseElement):
    def __init__(self,q,frame,density,S,x_integral,y_integral,x_f,q_f,EI,GJ,grav_vec=sym.Matrix([0]*3) ,simplify=True, name="default", K=0):
        self._EI = EI
        self._GJ = GJ
        self._K = K
        self._x_f = x_f
        self._x_integral = x_integral
        self._y_integral = y_integral

        self._q_f = sym.Matrix(q_f) if isinstance(q_f,Iterable) else sym.Matrix([q_f])
        self._S = S
        self._frame = frame

        if isinstance(frame,HomogenousFrame):
            q = q
        elif isinstance(frame,ReferenceFrame):
            q = sym.Matrix([*frame.R,*frame.theta,*q_f])
        else:
            raise ValueError('Invlaid Transform type')

        super(FlexiElement, self).__init__(q,name)

        self._grav_vec = sym.Matrix(grav_vec)
        self._density = density
        self._simplify = simplify

    x = property(lambda self: self.x_integral[0])
    y = property(lambda self: self.y_integral[0])
    EI = property(lambda self: self._EI)
    GJ = property(lambda self: self._GJ)
    K = property(lambda self: self._K)
    x_f = property(lambda self: self._x_f)
    x_integral = property(lambda self: self._x_integral)
    y_integral = property(lambda self: self._y_integral)
    q_f = property(lambda self: self._q_f)
    S = property(lambda self: self._S)
    grav_vec = property(lambda self: self._grav_vec)
    density = property(lambda self: self._density)
    simplify = property(lambda self: self._simplify)

    def _get_frame(self):
        return self._frame

    def _set_frame(self, value):
        self._frame = value

    frame = property(_get_frame, _set_frame)

    @property
    @cache
    def u_0(self):
        return sym.Matrix([self.x,self.y,0])

    @property
    @cache
    def u_f(self):
        return self.S*self.q_f

    @property
    @cache
    def ke(self):
        if self.density==0:
            return 0
        T = sym.Rational(1,2)*self.qd.T*self.M*self.qd
        return T[0].expand()

    @property
    @cache
    def pe(self):
        return self.elastic_pe + self.grav_pe

    @property
    @cache
    def elastic_pe(self):
        U_e = 0
        v_y = msubs(self.u_f,{self.x:self.x_f}).diff(self.y,self.y)
        v_theta = msubs(self.u_f.diff(self.x),{self.x:self.x_f}).diff(self.y)

        # Out-of-plane bending P.E per unit length
        if isinstance(self.EI, sym.Expr) or self.EI != 0:
            strain_ob = v_y[2]**2
            U_e += (sym.expand(strain_ob) if self.simplify else strain_ob)*self.EI*sym.Rational(1,2)

        # Torsional P.E per unit length
        if isinstance(self.GJ, sym.Expr) or self.GJ != 0:
            strain_t = (v_theta.T*v_theta)[0]
            U_e += (sym.expand(strain_t) if self.simplify else strain_t)*self.GJ*sym.Rational(1,2)

        # Bending-torsion cross coupling P.E per unit length
        if isinstance(self.K, sym.Expr) or self.K != 0:
            coupling = (v_y.T*v_theta)[0]
            coupling_e = sym.expand(coupling) if self.simplify else coupling
            U_e -= coupling_e*self.K*sym.Rational(1,2)
            U_e -= coupling_e*self.K*sym.Rational(1,2)

        return U_e.integrate(self.y_integral) if isinstance(U_e, sym.Expr) else U_e

    @property
    @cache
    def grav_pe(self):
        if (self.grav_vec.T*self.grav_vec)[0] == 0:
            return 0
        elif isinstance(self.frame,HomogenousFrame):
            point = self.frame.R
            dmg = -(point.T*self.grav_vec)[0]*self.density
            return dmg.integrate(self.x_integral,self.y_integral) if dmg != 0 else 0
        elif isinstance(self.frame,ReferenceFrame):
            g = -self.grav_vec
            mg = self.density.integrate(self.x_integral,self.y_integral)*self.frame.R.T*g
            mg += (self.density*(self.u_0 + self.u_f).T).integrate(self.x_integral,self.y_integral)*self.frame.A.T*g
            return mg[0]
        else:
            raise ValueError('Transform must either be type HomogenousFrame or ReferenceFrame')


    @property
    def rdf(self):
        return sym.Integer(0)

    @property
    @cache
    def M(self):
        if isinstance(self.frame,HomogenousFrame):
            Js = self.frame.ManipJacobian(self.q)
            Jb = self.frame.InvAdjoint()*Js
            if self.simplify:
                Jb = self._trigsimp(Jb)
            M_e = MassMatrix(self.density)
            M = (Jb.T*M_e*Jb).integrate(self.x_integral,self.y_integral)
        elif isinstance(self.frame,ReferenceFrame):
            M_rr = sym.eye(3)*(self.density.integrate(self.x_integral,self.y_integral))
            u = Wedge(self.u_0 + self.u_f)
            A = self.frame.A
            Gb = self.frame.Gb
            S = self.S

            M_rt = -A
            M_rt *= (self.density*u).integrate(self.x_integral,self.y_integral)
            M_rt *= Gb

            M_rf = A * (self.density * S).integrate(self.x_integral,self.y_integral)
            M_tt = Gb.T
            M_tt *= (self.density*u.T*u).integrate(self.x_integral,self.y_integral)
            M_tt *= Gb

            M_tf = Gb.T
            M_tf *= (self.density*u*S).integrate(self.x_integral,self.y_integral)

            M_ff = (self.density*S.T*S).integrate(self.x_integral,self.y_integral)

            M = sym.BlockMatrix([[M_rr,M_rt,M_rf],[M_rt.T,M_tt,M_tf],[M_rf.T,M_tf.T,M_ff]]).as_explicit()
        else:
            raise ValueError("Transform must either be type HomogenousFrame or ReferenceFrame")
        return self._trigsimp(M) if self.simplify else M

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.expand(expr))

    @staticmethod
    def ShapeFunctions_OBM_TO(m, o, q, y_s, x_s, x_f, factor=1, type='taylor', return_slopes=False):
        """Returns shape function matrix for a beam with out-of-plane bending and torsion.

        Parameters
        ----------
        m : int
            number of out-of-plane bending shapes
        o : int
            number of torsion shapes
        q : list
            list of DoFs in order [bending..., torsion...]
        y_s : Symbol
            spanwise coordinate (beam integrates along y)
        x_s : Symbol
            chordwise coordinate
        x_f : Symbol
            flexural axis chordwise location (twist = 0 at x_s = x_f)
        factor : int / float / list
            scaling factor per DoF (scalar applies to all)
        type : str
            'taylor' or 'cheb' polynomial basis
        return_slopes : bool
            if True, also returns dS/dx and dS/dy slope matrices

        Returns
        -------
        S : Matrix (3 x len(q))
        tau : Expr   (twist distribution)
        S_x, S_y : Matrix (optional, only if return_slopes=True)
        """
        q = q if isinstance(q,Iterable) else [q]
        _x = sym.Dummy('x_s')
        _y = sym.Dummy('y_s')

        qs = len(q)
        if m + o != qs:
            raise ValueError('m+o must equal len(q)')
        S = sym.zeros(3, qs)

        if isinstance(factor, int) or isinstance(factor, float):
            factor = [factor]*qs

        # Out-of-plane bending: z-displacement
        # Taylor:    y^(2+i)              — satisfies w(0)=0, w'(0)=0
        # Chebyshev: y^2 * T_i(2y-1)     — maps domain [0,1]→[-1,1] and satisfies both root BCs
        for i in range(m):
            if type == 'taylor':
                S[2,i] += _y**(2+i)*factor[i]
            elif type == 'cheb':
                S[2,i] += _y**2 * sym.chebyshevt_poly(i, 2*_y - 1)*factor[i]
            else:
                raise ValueError("type must be 'taylor' or 'cheb'")

        # Torsion: z-displacement with chordwise offset from flexural axis
        # Taylor:    y^(i+1)              — satisfies tau(0)=0
        # Chebyshev: y * T_i(2y-1)       — satisfies tau(0)=0, correct domain mapping
        tau = sym.Integer(0)
        for i in range(o):
            qi = i + m
            if type == 'taylor':
                S[2,qi] += -_y**(i+1)*factor[qi]*(_x - x_f)
                tau += q[qi]*_y**(i+1)*factor[qi]
            elif type == 'cheb':
                S[2,qi] += -_y * sym.chebyshevt_poly(i, 2*_y - 1)*factor[qi]*(_x - x_f)
                tau += q[qi]*_y * sym.chebyshevt_poly(i, 2*_y - 1)*factor[qi]

        if return_slopes:
            S_x = S.diff(_x).subs({_x: x_s, _y: y_s})
            S_y = S.diff(_y).subs({_x: x_s, _y: y_s})
            S = S.subs({_x: x_s, _y: y_s})
            tau = tau.subs({_x: x_s, _y: y_s})
            if type == 'cheb':
                S_x = sym.expand(S_x)
                S_y = sym.expand(S_y)
                S = sym.expand(S)
                tau = sym.expand(tau)
            return S, tau, S_x, S_y

        S = S.subs({_x: x_s, _y: y_s})
        tau = tau.subs({_x: x_s, _y: y_s})
        if type == 'cheb':
            S = sym.expand(S)
            tau = sym.expand(tau)
        return S, tau

    @staticmethod
    def ShapeFunctions_OBM_IBN_TO(m, n, o, q, y_s, x_s, x_f, factor=1, type='taylor', return_slopes=False):
        """Deprecated — in-plane bending (IBN) has been removed. Use ShapeFunctions_OBM_TO."""
        import warnings
        if n != 0:
            raise ValueError(
                "In-plane bending (n != 0) is no longer supported. "
                "Use ShapeFunctions_OBM_TO(m, o, ...) instead."
            )
        warnings.warn(
            "ShapeFunctions_OBM_IBN_TO is deprecated; use ShapeFunctions_OBM_TO(m, o, ...).",
            DeprecationWarning, stacklevel=2
        )
        return FlexiElement.ShapeFunctions_OBM_TO(m, o, q, y_s, x_s, x_f, factor, type, return_slopes)
