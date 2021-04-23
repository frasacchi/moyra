import sympy as sym
from .base_element import BaseElement
from sympy.physics.mechanics import msubs

class FlexiElement(BaseElement):
    def __init__(self,Transform,M,x,y,z,c,s,x_f,EI,GJ,gravityPot = False):
        
        self.x = x
        self.y = y
        self.z = z
        self.c = c
        self.s = s
        self.EI = EI
        self.GJ = GJ
        self.x_f = x_f

        self.y_integral = (self.y,0,self.s)
        self.x_integral = (self.x,0,self.c)

        self._gravityPotential = gravityPot

        self.Transform = Transform
        self.dTransform = Transform.Translate(self.x,self.y,self.z)
        self.M_e = M

    def calc_ke(self, p):
        M = self.M(p)
        # calculate the K.E
        T = sym.Rational(1,2)*p.qd.T*(M.integrate(self.x_integral,self.y_integral))*p.qd
        return T[0].expand()

    
    def M(self,p):
        # create the jacobian for the mass    
        Js = self.dTransform.ManipJacobian(p.q)
        Jb = self.dTransform.InvAdjoint()*Js
        Jb = self._trigsimp(Jb)
        #calculate the mass Matrix in world frame
        return Jb.T*self.M_e*Jb

    def _trigsimp(self,expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))


    def calc_elastic_pe(self,p):
        # Bending Potential Energy per unit length
        v = self.dTransform.msubs({self.x:self.x_f}).diff(self.y,self.y).Transform_point([0,0,0])
        U_e = self._trigsimp((v.T*v))[0]*self.EI*sym.Rational(1,2)

        # Torsional P.E per unit length
        v = self.dTransform.diff(self.x).diff(self.y).Transform_point([self.x_f,0,0])
        U_e += self._trigsimp((v.T*v))[0]*self.GJ*sym.Rational(1,2)

        return U_e.integrate(self.y_integral)

    def calc_grav_pe(self, p):
        point_z = self.dTransform.Transform_point([0]*3)[2]
        dmg = point_z*self.M_e[0,0]*p.g
        return dmg.integrate(self.x_integral,self.y_integral)

    def calc_pe(self,p):
        PE = self.calc_elastic_pe(p)
        PE += self.calc_grav_pe(p) if self._gravityPotential else sym.Integer(0)
        return PE

    def calc_rdf(self,p):
        return 0

    @staticmethod
    def ShapeFunctions_BN_TM(n,m,q,y_s,x,x_f,alpha_r,factor = 1):
        # check q is the length of n+m
        if n+m != len(q):
            raise ValueError('the sum of n+m must be the same as a length of q')

        # make factor a list the size of n+m
        if isinstance(factor,int) | isinstance(factor,float):
            factor = [factor]*(n+m)
        z = sym.Integer(0)
        tau = alpha_r

        for i in range(0,n):
            z = z + q[i]*y_s**(2+i)*factor[i]
        for i in range(0,m):
            qi = i+n
            tau = tau + q[qi]*y_s**(i+1)*factor[n+i]
        
        z -= tau*(x-x_f)

        return z, tau




            
