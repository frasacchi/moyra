import sympy as sym
from .base_element import BaseElement
from sympy.physics.mechanics import msubs,dot

class FlexiElement(BaseElement):
    def __init__(self,Transform,M,x,y,x_integral,y_integral,x_f,EI,GJ,gravityPot = False):
        
        self.x = x
        self.y = y
        self.EI = EI
        self.GJ = GJ
        self.x_f = x_f
        self.x_integral = x_integral
        self.y_integral = y_integral

        self._gravityPotential = gravityPot

        self.Transform = Transform
        self.M_e = M

    def calc_ke(self, p):
        if self.M_e.trace()==0:
            return 0
        M = self.M(p)
        # calculate the K.E
        T = sym.Rational(1,2)*p.qd.T*(M.integrate(self.x_integral,self.y_integral))*p.qd
        return T[0].expand()

    
    def M(self,p):
        # create the jacobian for the mass    
        Js = self.Transform.ManipJacobian(p.q)
        Jb = self.Transform.InvAdjoint()*Js
        Jb = self._trigsimp(Jb)
        #calculate the mass Matrix in world frame
        return Jb.T*self.M_e*Jb

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))


    def calc_elastic_pe(self,p):
        # Bending Potential Energy per unit length
        U_e = 0
        if isinstance(self.EI, sym.Expr) or self.EI != 0:
            v = msubs(self.Transform.t,{self.x:self.x_f}).diff(self.y,self.y)
            U_e += self._trigsimp((v.T*v))[0]*self.EI*sym.Rational(1,2)

        # Torsional P.E per unit length
        if isinstance(self.GJ, sym.Expr) or self.GJ != 0:
            v = msubs(self.Transform.t.diff(self.x,self.y),{self.x:self.x_f})
            U_e += self._trigsimp((v.T*v))[0]*self.GJ*sym.Rational(1,2)

        return U_e.integrate(self.y_integral) if isinstance(U_e, sym.Expr) else U_e

    def calc_grav_pe(self, p):
        point = self.Transform.t
        h = -(point.T*p.g_v)[0]
        dmg = h*self.M_e[0,0]*p.g
        return dmg.integrate(self.x_integral,self.y_integral)

    def calc_pe(self,p):
        PE = self.calc_elastic_pe(p)
        PE += self.calc_grav_pe(p) if self._gravityPotential else sym.Integer(0)
        return PE

    def calc_rdf(self,p):
        return 0

    @staticmethod
    def ShapeFunctions_BN_TM(n,m,q,y_s,x,x_f,alpha_r,factor = 1,type='taylor'):
        # check q is the length of n+m
        if n+m != len(q):
            raise ValueError('the sum of n+m must be the same as a length of q')
            

        # make factor a list the size of n+m
        if isinstance(factor,int) | isinstance(factor,float):
            factor = [factor]*(n+m)
        z = sym.Integer(0)
        tau = alpha_r
        for i in range(0,n):
            if type == 'taylor':
                z += q[i]*y_s**(2+i)*factor[i]
            elif type == 'cheb':
                z += q[i]*sym.chebyshevt_poly(i,y_s)*factor[i]
            else:
                raise ValueError('poly type must be either cheb or taylor')
        for i in range(0,m):
            qi = i+n
            if type == 'taylor':
                tau += q[qi]*y_s**(i+1)*factor[n+i]
            elif type == 'cheb':
                tau += q[qi]*sym.chebyshevt_poly(i,y_s)*factor[i]
        z -= tau*(x-x_f)

        return sym.simplify(z), sym.simplify(tau)




            
