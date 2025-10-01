import sympy as sym
from . import ExternalForce
from ..helper_funcs import linearise_matrix
import sympy.physics.mechanics as me

class AeroForce(ExternalForce):
    """A Class used to represent aerodynamic forces"""
    @classmethod
    def PerUnitSpan(cls,p,Transform,C_L,alphadot,M_thetadot,e,alpha_zero=0, root_alpha=0, delta_alpha=0,stall_angle=0.24,c_d_max = 1,w_g = 0,V=None,c=None,linear=False,z_inverted=False,c_l_func=None,C_M=0,c_m_func=None,unsteady=False):
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
        (Q,dAlpha) = cls.get_Q(p,Transform,C_L,alphadot,M_thetadot,e,
                            alpha_zero,root_alpha,delta_alpha,stall_angle,c_d_max,w_g,
                            V,c,linear,z_inverted,c_l_func,C_M,c_m_func,unsteady)
        return cls(Q,dAlpha)
        
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
    def get_dAlpha(p,BodyJacobian,alphadot,alpha_zero=0, root_alpha=0, delta_alpha=0, w_g = 0,V=None,c=None,z_inverted=False,unsteady=False):
        if c is None:
            c=p.c

        v_z_eff = (BodyJacobian*p.qd)[2]
        if z_inverted:
            v_z_eff *= -1
        if V is None:
            V = -(BodyJacobian*p.qd)[0]
        
        alpha = delta_alpha + sym.atan(-v_z_eff/V) + w_g/V
        if unsteady:
            c1 = 0.5*p.AR/(2.32+p.AR)
            c2 = 0.181+0.772/p.AR
            G = -c1*c2*p.k/(p.k**2+c2**2)
            F = 1-c1*p.k**2/(p.k**2+c2**2)
            lag_alpha =  p.AR/(2+p.AR)*(F*(alpha)+p.c/(2*V*p.k)*G*alphadot)
        else:
            lag_alpha =  alpha
        return alpha_zero + root_alpha + lag_alpha

    def get_apparent_mass(p,BodyJacobian,C_L,V=None,c=None,linear=False,c_l_func=None):
        """
        see the class method PerUnitSpan for a explaination of terms
        """
        if c is None:
            c=p.c
        if V is None:
            V = -(BodyJacobian*p.qd)[0]
        v = (BodyJacobian*p.qd)[2]
        return p.rho*C_L*p.c**2*v.diff(me.dynamicsymbols._t)/8

    def get_F_n(p,BodyJacobian,dAlpha,C_L,alpha_zero=0, stall_angle=0.24,c_d_max = 1,V=None,c=None,linear=False,c_l_func=None):
        """
        see the class method PerUnitSpan for a explaination of terms
        """
        if c is None:
            c=p.c
        if V is None:
            V = -(BodyJacobian*p.qd)[0]

        v_z_eff = (BodyJacobian*p.qd)[2]
        V_rel = sym.sqrt(v_z_eff**2 + V**2)
        # Calculate the lift force
        dynamicPressure = sym.Rational(1,2)*p.rho*V_rel**2

        # Calculate C_L curve
        if c_l_func is not None:
            c_l = c_l_func(dAlpha)
        else:
            if stall_angle == 0:
                c_l = C_L*dAlpha
            else:
                c_l = C_L*(1/p.clip_factor*sym.ln((1+sym.exp(p.clip_factor*(dAlpha+stall_angle)))/(1+sym.exp(p.clip_factor*(dAlpha-stall_angle))))-stall_angle)

        if linear:
            c_n = c_l
        else:
            c_d = c_d_max*sym.Rational(1,2)*(1-sym.cos(2*dAlpha))
            ang = dAlpha - alpha_zero
            c_n = c_l*sym.cos(ang)+c_d*sym.sin(ang)

        ## joint torques for lift are calculated in a frame aligned with the chordwise velocity direction
        return  dynamicPressure*c*c_n
        
    def get_moment(p,BodyJacobian,dAlpha,F_n,C_L,alphadot,M_thetadot,e,V=None,c=None,C_M=0,c_m_func=None,unsteady=False):
        if c_m_func is not None:
            c_m = c_m_func(dAlpha)
        else:
            c_m = C_M*dAlpha
        if V is None:
            V = -(BodyJacobian*p.qd)[0]   
        dynamicPressure = sym.Rational(1,2)*p.rho*V**2
        M_w = -F_n*e*c + dynamicPressure*c*-c_m# Moment due to lift
        M_w += dynamicPressure*c**2*(M_thetadot*alphadot*c/(sym.Integer(4)*V))
        return M_w

    def get_Q(p,Transform,C_L,alphadot,M_thetadot,e,alpha_zero=0, root_alpha=0, delta_alpha=0, stall_angle=0.24,c_d_max = 1,w_g = 0,V=None,c=None,linear=False,z_inverted=False,c_l_func=None,C_M=0,c_m_func=None,unsteady=False):
        """
        see the class method PerUnitSpan for a explaination of terms
        """
        BodyJacobian = sym.simplify(AeroForce._trigsimp(Transform.BodyJacobian(p.q)))
        dAlpha = AeroForce.get_dAlpha(p,BodyJacobian,alphadot,alpha_zero, root_alpha, delta_alpha, w_g,V,c,z_inverted,unsteady)
        F_n = AeroForce.get_F_n(p,BodyJacobian,dAlpha,C_L,alpha_zero, stall_angle,c_d_max,V,c,linear,c_l_func)
        M_w = AeroForce.get_moment(p,BodyJacobian,dAlpha,F_n,C_L,alphadot,M_thetadot,e,V,c,C_M,c_m_func,unsteady)
        if c is None:
            c=p.c
        if V is None:
            V = -(BodyJacobian*p.qd)[0]
        # Calculate the lift force

        ## joint torques for lift are calculated in a frame aligned with the chordwise velocity direction
        wrench_lift = sym.Matrix([0,0,-F_n,0,0,0]) if z_inverted else sym.Matrix([0,0,F_n,0,0,0])
        Q_L = BodyJacobian.T*wrench_lift

        ## joint torques for moment are calculated in a frame aligned with the shord of the wing
        BodyJacobian = sym.simplify(AeroForce._trigsimp(Transform.R_y(delta_alpha).BodyJacobian(p.q)))
        wrench_moment = sym.Matrix([0,0,0,0,M_w,0])
        Q_M = BodyJacobian.T*wrench_moment

        # combine forces
        Q = Q_L + Q_M
        return (Q,dAlpha)


