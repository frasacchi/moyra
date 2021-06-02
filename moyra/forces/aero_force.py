import sympy as sym
from . import ExternalForce
from ..helper_funcs import linearise_matrix
import sympy.physics.mechanics as me

class AeroForce(ExternalForce):
    """A Class used to represent aerodynamic forces"""
    @classmethod
    def PerUnitSpan(cls,p,Transform,C_L,alphadot,M_thetadot,e,rootAlpha,alpha_zero = 0,stall_angle=0.24,c_d_max = 1,w_g = 0,V=None,c=None,linear=False,z_inverted=False):
        """ A static method to create an aerodynamic force per unit span

        ...
        Parameters
        ----------
        p - an instance of a 'ModelParameters' class
        Transform - an instance of a 'HomogenousTransform' class describing the coordinate system the force is applied in
        C_L - the 2D lift curve slope
        alphadot - unsteady aerodynamic coeffiecent (see Cooper & Wright)
        M_thetadot - unsteady aerodynamic coeffiecent (see Cooper & Wright)
        e - chord wing eccentrity
        rootAlpha - root angle of attack of the element
        alpha_zero - angle attack for zero lift (camber of aerofoil)
        
        Optional Parameters
        -------------------

        stall_angle - angle in radians at which the wing stalls, defining the lift curve slope. If zero assume linear (default = 0)
        c_d_max - max coeefgficent of drag (default = 1)
        w_g - gust velocity (default = 0)
        V - onset wind velocity (default p.V)
        c - chord (default p.c)
        linear - if true normal force is a combination of lift and drag (default false)
        z_inverted - if true lift acts in the oppiste direction to the z axis
        """
        ## force per unit length will following theredosons pseado-steady theory

        BodyJacobian = sym.simplify(cls._trigsimp(Transform.BodyJacobian(p.q)))

        (wrench,dAlpha) = cls.get_wrench(p,BodyJacobian,C_L,alphadot,M_thetadot,e,
                            rootAlpha,alpha_zero,stall_angle,c_d_max,w_g,
                            V,c,linear,z_inverted)

        _Q = BodyJacobian.T*wrench

        return cls(_Q,dAlpha)
        
    def __init__(self,Q,dAlpha):
        self.dAlpha = dAlpha
        super().__init__(Q) 

    @staticmethod
    def _trigsimp(expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))      

    def linearise(self,x,x_f):
        Q_lin = linearise_matrix(self.Q(),x,x_f)
        dAlpha_lin = linearise_matrix(self.dAlpha,x,x_f)
        return AeroForce(Q_lin,dAlpha_lin)
    
    def subs(self,*args):
        return AeroForce(self._Q.subs(*args),self.dAlpha.subs(*args))

    def msubs(self,*args):
        return AeroForce(me.msubs(self._Q,*args),me.msubs(self.dAlpha,*args))

    def integrate(self,*args):
        return AeroForce(self._Q.integrate(*args),self.dAlpha)

    @staticmethod
    def get_wrench(p,BodyJacobian,C_L,alphadot,M_thetadot,e,rootAlpha,alpha_zero = 0,stall_angle=0.24,c_d_max = 1,w_g = 0,V=None,c=None,linear=False,z_inverted=False):
        """
        see the class method PerUnitSpan for a explaination of terms
        """
        if c is None:
            c=p.c

        v_z_eff = (BodyJacobian*p.qd)[2]
        if z_inverted:
            v_z_eff *= -1
        if V is None:
            V = -(BodyJacobian*p.qd)[0]
        
        # combine to get effective AoA
        dAlpha = alpha_zero + rootAlpha - v_z_eff/V + w_g/V

        # Calculate the lift force
        dynamicPressure = sym.Rational(1,2)*p.rho*V**2

        # Calculate C_L curve
        if stall_angle == 0:
            c_l = C_L*dAlpha
        else:
            c_l = C_L*(1/p.clip_factor*sym.ln((1+sym.exp(p.clip_factor*(dAlpha+stall_angle)))/(1+sym.exp(p.clip_factor*(dAlpha-stall_angle))))-stall_angle)

        if linear:
            c_n = c_l
        else:
            c_d = c_d_max*sym.Rational(1,2)*(1-sym.cos(2*dAlpha))
            ang = rootAlpha - v_z_eff/V
            c_n = c_l*sym.cos(ang)+c_d*sym.sin(ang)

        F_n = dynamicPressure*c*c_n

        # Calulate the pitching Moment
        M_w = F_n*e*c # Moment due to lift
        M_w += dynamicPressure*c**2*(M_thetadot*alphadot*c/(sym.Integer(4)*V))
        
        if z_inverted:
            wrench = sym.Matrix([0,0,-F_n,0,M_w,0])
        else:
            wrench = sym.Matrix([0,0,F_n,0,M_w,0])
        return (wrench,dAlpha)





