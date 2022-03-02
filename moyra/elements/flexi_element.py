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
    def __init__(self,Transform,density,S,x_integral,y_integral,x_f,q_f,EI,GJ,gravityPot = False,simplify=True):
        
        self.x = x_integral[0]
        self.y = y_integral[0]
        self.EI = EI
        self.GJ = GJ
        self.x_f = x_f
        self.x_integral = x_integral
        self.y_integral = y_integral

        self.q_f = q_f if isinstance(q_f,Iterable) else [q_f]
        self.u_0 = sym.Matrix([self.x,self.y,0])
        self.S = S
        self.u_f = S*q_f

        self._gravityPotential = gravityPot

        self.Transform = Transform
        self.density = density
        self.simplify = simplify

    def q(self,p):
        if isinstance(self.Transform,HomogenousFrame):   
            return p.q
        if isinstance(self.Transform,ReferenceFrame):  
            return sym.Matrix([*self.Transform.R,*self.Transform.theta,*self.q_f])
        raise ValueError('Transform must either be type HomogenousFrame or ReferenceFrame')
      
    def qd(self,p):
        return self.q(p).diff(t)

    def calc_ke(self, p, M = None):
        if self.density==0:
            return 0
        M = M if M is not None else self.M(p)
        # calculate the K.E
        T = sym.Rational(1,2)*self.qd(p).T*M*self.qd(p)
        return T[0].expand()

    
    def M(self,p):
        # create the jacobian for the mass 
        
        if isinstance(self.Transform,HomogenousFrame):   
            Js = self.Transform.ManipJacobian(p.q)
            Jb = self.Transform.InvAdjoint()*Js
            Jb = self._trigsimp(Jb)
            M_e = MassMatrix(self.density)
            M = (Jb.T*M_e*Jb).integrate(self.x_integral,self.y_integral)
        elif isinstance(self.Transform,ReferenceFrame):
            M_rr = sym.eye(3)*(self.density.integrate(self.x_integral,self.y_integral))
            u = Wedge(self.u_0 + self.u_f)
            A = self.Transform.A

            M_rt = -A
            M_rt *= (self.density*u).integrate(self.x_integral,self.y_integral)
            M_rt *= self.Transform.Gb

            M_rf = A * (self.density * self.S).integrate(self.x_integral,self.y_integral)
            M_tt = self.Transform.Gb.T
            M_tt *= (self.density*u.T*u).integrate(self.x_integral,self.y_integral)
            M_tt *= self.Transform.Gb

            M_tf = self.Transform.Gb.T
            M_tf *= (self.density*u*self.S).integrate(self.x_integral,self.y_integral)

            M_ff = (self.density*self.S.T*self.S).integrate(self.x_integral,self.y_integral)

            M = sym.BlockMatrix([[M_rr,M_rt,M_rf],[M_rt.T,M_tt,M_tf],[M_rf.T,M_tf.T,M_ff]]).as_explicit()
        else:
            raise ValueError("Transform must either be type HomogenousFrame or ReferenceFrame")
        return self._trigsimp(M) if self.simplify else M

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))


    def calc_elastic_pe(self,p):
        # Bending Potential Energy per unit length
        U_e = 0
        if isinstance(self.EI, sym.Expr) or self.EI != 0:
            v = msubs(self.u_f,{self.x:self.x_f}).diff(self.y,self.y)
            U_e += self._trigsimp((v.T*v))[0]*self.EI*sym.Rational(1,2)

        # Torsional P.E per unit length
        if isinstance(self.GJ, sym.Expr) or self.GJ != 0:
            v = msubs(self.u_f.diff(self.x),{self.x:self.x_f}).diff(self.y)
            U_e += self._trigsimp((v.T*v))[0]*self.GJ*sym.Rational(1,2)

        return U_e.integrate(self.y_integral) if isinstance(U_e, sym.Expr) else U_e

    def calc_grav_pe(self, p):
        if isinstance(self.Transform,HomogenousFrame):   
            point = self.Transform.R
            h = -(point.T*p.g_v)[0]
            dmg = h*self.density*p.g
            return dmg.integrate(self.x_integral,self.y_integral)
        if isinstance(self.Transform,ReferenceFrame):  
            g = -p.g_v * p.g
            mg = self.density.integrate(self.x_integral,self.y_integral)*self.Transform.R.T*g
            mg += (self.density*(self.u_0 + self.u_f).T).integrate(self.x_integral,self.y_integral)*self.Transform.A.T*g
            return mg[0]
        raise ValueError('Transform must either be type HomogenousFrame or ReferenceFrame')

    def calc_pe(self,p):
        PE = self.calc_elastic_pe(p)
        PE += self.calc_grav_pe(p) if self._gravityPotential else sym.Integer(0)
        return PE

    def calc_rdf(self,p):
        return sym.Integer(0)

    @staticmethod
    def ShapeFunctions_OBM_IBN_TO(m,n,o,q,y_s,x_s,x_f,factor = 1,type='taylor'):
        """
        returns the shape function mAtrix for an assumed shapes beam
        
        The function returns a tple of the shape function matrix S and the twist
         of an assumed shapes beam with 'm' out-of-plane bending shpaes, 'n' 
         in-plane bending shapes, and 'o' twist shapes, the list 'q' represents the DoF in the order n,m,o 
        
        Parameters
        ----------
        m : int
            number of out-of-plane bending shapes
        n : int
            number of in-plane bending shapes
        o : int 
        number of twist shapes
        q : list
            list of DoFs
        y_s : Symbol
            the y direction of the beam (the beam is integrated along its y direction)
        x_s : Symbol
            the x direction of the beam (XY plane is in plane bending)
        x_f : Symbol
            the location of the flexural axis e.g twist = 0 @ x_s=x_f
        factor : int / float / list
            scaling factor to apply to each DoF. If it is a single number
            it is applied to all DoFs, otherwise it must be a list equal in
            length to q
        type : str
            either 'taylor' or 'cheb' and defines the shape functions used 
            (talyor series or chebeshev series)

        Returns
        -------
        S:Dense Matrix
            A 3xlen(q) Matrix which defines the shape functions. Sq = v where v 
            is a 3x1 vector defining the deformed shape at a given undeformed 
            x_s and y_s position
        tau: Sympy Function
            a function defining the twist along the flexural axis
         """
        # ensure q is iterable even if only one element
        q = q if isinstance(q,Iterable) else [q]

        # check elements in q the same as n+m+o
        qs = len(q)
        if n+m+o != qs:
            raise ValueError('the sum of n+m+o must be the same as a length of q')
        S = sym.zeros(3,qs)
        # make factor a list the size of n+m
        if isinstance(factor,int) | isinstance(factor,float):
            factor = [factor]*(qs)

        # Out of plane bending
        for i in range(0,m):
            if type == 'taylor':
                S[2,i] += y_s**(2+i)*factor[i]
            elif type == 'cheb':
                S[2,i] += sym.chebyshevt_poly(i,y_s)*factor[i]
            else:
                raise ValueError('poly type must be either cheb or taylor')
        # In-plane bending
        for i in range(0,n):
            if type == 'taylor':
                S[0,i] += y_s**(2+i)*factor[i]
            elif type == 'cheb':
                S[0,i] += sym.chebyshevt_poly(i,y_s)*factor[i]
            else:
                raise ValueError('poly type must be either cheb or taylor')
        # Twist
        tau = 0
        for i in range(0,o):
            qi = i+n
            if type == 'taylor':
                S[2,qi] += -y_s**(i+1)*factor[n+i]*(x_s-x_f)
                tau += q[qi]*y_s**(i+1)*factor[n+i]
            elif type == 'cheb':
                S[2,qi] += -sym.chebyshevt_poly(i,y_s)*factor[i]*(x_s-x_f)
                tau += q[qi]*sym.chebyshevt_poly(i,y_s)*factor[i]

        return sym.simplify(S), sym.simplify(tau)




            
