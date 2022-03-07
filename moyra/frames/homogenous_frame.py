from functools import cache
from moyra.frames.base_reference_frame import BaseReferenceFrame
import sympy as sym
from sympy.abc import t
import sympy.physics.mechanics as me
from moyra.helper_funcs import Vee, Vee4, Wedge, Wedge4


class HomogenousFrame(BaseReferenceFrame):

    def __init__(self,E=None):
        E = sym.eye(4) if E is None else E
        super(HomogenousFrame, self).__init__(E[:3,:3],E[:3,3])
    
    @property
    @cache
    def E(self):
        E = sym.eye(4)
        E[:3,:3] = self.A
        E[:3,3] = self.R
        return E


    def BodyJacobian(self,q):
        return self.InvAdjoint()*self.ManipJacobian(q)

    def __mul__(self,other):
        if isinstance(other,HomogenousFrame):
            return HomogenousFrame(self.E*other.E)  
        elif isinstance(other,sym.MutableDenseMatrix):
            return HomogenousFrame(self.E*other)
        else:
            raise TypeError(f'Can not multiple a Homogenous Transform by type {type(other)}')

    def Inverse(self):
        E = sym.eye(4)
        E[:3,:3] = self.A.T
        E[:3,3] = -self.A.T*self.R
        return HomogenousFrame(E)

    def ManipJacobian(self,q):
        inv = self.Inverse().E
        J = sym.zeros(6,len(q))
        for i,qi in enumerate(q):
            J[:,i] = Vee4(self.E.diff(qi)*inv)
        return J

    def Adjoint(self):
        E = sym.zeros(6,6)
        E[:3,:3] = self.A
        E[3:,3:] = self.A
        E[:3,3:] = Wedge(self.R)*self.A.T
        return E

    def InvAdjoint(self):
        E = sym.zeros(6,6)
        E[:3,:3] = self.A.T
        E[3:,3:] = self.A.T
        E[:3,3:] = -self.A.T*Wedge(self.R)
        return E

    def Diff(self):
        E = sym.eye(4)
        E[:3,:3] = self.A.diff(t)
        E[:3,3] = self.R.diff(t)
        return E

    def BodyVelocity(self):
        V = sym.ones(6,1)
        V[:3,0] = self.A.T*self.R.diff(t)

        # Angular velocities skew symetric matrix
        S = self.A.T*self.A.diff(t)
        V[3,0] = S[2,1]
        V[4,0] = S[0,2]
        V[5,0] = S[1,0]
        return V

    def SpatialVelocity(self):
        return self.Diff()*self.Inverse()

    def PuesdoSpatialFrame(self):
        E = self.E.copy()
        E[:3,:3] = sym.eye(3)
        return HomogenousFrame(self.q, E)
    
    def Rotate(self,A):
        H = sym.eye(4)
        H[:3,:3]=A
        return HomogenousFrame(self.E*H)

    def R_x(self,angle):
        H = sym.eye(4)
        H[:3,:3]=sym.rot_axis1(-angle)
        return HomogenousFrame(self.E*H)

    def R_y(self,angle):
        H = sym.eye(4)
        H[:3,:3]=sym.rot_axis2(-angle)
        return HomogenousFrame(self.E*H)

    def R_z(self,angle):
        H = sym.eye(4)
        H[:3,:3]=sym.rot_axis3(-angle)
        return HomogenousFrame(self.E*H)

    def R_rodriguez_params(self,qs):
        H = sym.eye(4)
        qs = sym.Matrix(qs)
        R = sym.simplify(sym.eye(3)+2/(1+qs.dot(qs))*(Wedge(qs)+Wedge(qs)*Wedge(qs)))
        H[:3,:3] = R
        return HomogenousFrame(self.E*H)

    def R_euler_params(self,qs):
        H = sym.eye(4)
        qs = sym.Matrix(qs)
        E = sym.Matrix([[-qs[1],qs[0],-qs[3],qs[2]],[-qs[2],qs[3],qs[0],-qs[1]],[-qs[3],-qs[2],qs[1],qs[0]]])
        G = sym.Matrix([[-qs[1],qs[0],qs[3],-qs[2]],[-qs[2],-qs[3],qs[0],qs[1]],[-qs[3],qs[2],-qs[1],qs[0]]])
        H[:3,:3] = E*G.T
        return HomogenousFrame(self.E*H)

    def Translate(self,x,y,z):
        H = sym.eye(4)
        H[:3,3] = sym.Matrix([x,y,z])
        return HomogenousFrame(self.E*H)

    def simplify(self):
        return HomogenousFrame(sym.simplify(self.E))

    def diff(self,*args):  
        return HomogenousFrame(self.E.diff(*args))

    def subs(self,*args):  
        return HomogenousFrame(self.E.subs(*args))

    def msubs(self,*args):  
        return HomogenousFrame(me.msubs(self.E,*args))
