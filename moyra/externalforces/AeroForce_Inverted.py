import sympy as sym
from . import ExternalForce
from ..helper_funcs import LineariseMatrix
import sympy.physics.mechanics as me

class AeroForce_Inverted(ExternalForce):
    @classmethod
    def PerUnitSpan(cls,FwtParams,Transform,C_L,alphadot,M_thetadot,e,rootAlpha,alpha_zero = 0,stall_angle=0.24,c_d_max = 1,w_g = 0,V=None,c=None,linear=False):

        p = FwtParams
        ## force per unit length will following theredosons pseado-steady theory

        if V is None:
            V = p.V
        if c is None:
            c=p.c

        # add z velocity due to motion
        BodyJacobian = cls._trigsimp(Transform.BodyJacobian(p.q))

        v_z_eff = (BodyJacobian*p.qd)[2]
        
        # combine to get effective AoA
        dAlpha = alpha_zero + rootAlpha + v_z_eff/V + w_g/V

        # Calculate the lift force
        dynamicPressure = sym.Rational(1,2)*p.rho*V**2

        # Calculate C_L curve
        if stall_angle == 0:
            c_l = C_L*dAlpha
        else:
            c_l = C_L*(1/p.clip_factor*sym.ln((1+sym.exp(p.clip_factor*(dAlpha+stall_angle)))/(1+sym.exp(p.clip_factor*(dAlpha-stall_angle))))-stall_angle)

        c_d = c_d_max*sym.Rational(1,2)*(1-sym.cos(2*dAlpha))

        ang = rootAlpha + v_z_eff/V

        if linear:
            c_n = c_l
        else:
            c_n = c_l*sym.cos(ang)+c_d*sym.sin(ang)

        F_n = -dynamicPressure*c*c_n

        # Calulate the pitching Moment
        M_w = -F_n*e*c # Moment due to lift
        M_w += dynamicPressure*c**2*(M_thetadot*alphadot*c/(sym.Integer(4)*V))

        wrench = sym.Matrix([0,0,F_n,0,M_w,0])

        _Q = BodyJacobian.T*wrench

        return cls(_Q,dAlpha)
        
    def __init__(self,Q,dAlpha):
        self.dAlpha = dAlpha
        super().__init__(Q) 

    @staticmethod
    def _trigsimp(expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))      

    def linearise(self,x,x_f):
        Q_lin = LineariseMatrix(self.Q(),x,x_f)
        dAlpha_lin = LineariseMatrix(self.dAlpha,x,x_f)
        return AeroForce_Inverted(Q_lin,dAlpha_lin)
    
    def subs(self,*args):
        return AeroForce_Inverted(self._Q.subs(*args),self.dAlpha.subs(*args))

    def msubs(self,*args):
        return AeroForce_Inverted(me.msubs(self._Q,*args),me.msubs(self.dAlpha,*args))

    def integrate(self,*args):
        return AeroForce_Inverted(self._Q.integrate(*args),self.dAlpha)




