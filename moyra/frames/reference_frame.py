from moyra.frames.base_reference_frame import BaseReferenceFrame
import sympy as sym
from sympy.abc import t
import sympy.physics.mechanics as me

from moyra.helper_funcs import Wedge


class ReferenceFrame(BaseReferenceFrame):

    @classmethod
    def EulerParameters(cls,R,theta):
        G = sym.Integer(2)*sym.Matrix([[-theta[1],theta[0],-theta[3],theta[2],],
                                        [-theta[2],theta[3],theta[0],-theta[1],],
                                        [-theta[3],-theta[2],theta[1],theta[0],]])
        Gb = sym.Integer(2)*sym.Matrix([[-theta[1],theta[0],theta[3],-theta[2],],
                                        [-theta[2],-theta[3],theta[0],theta[1],],
                                        [-theta[3],theta[2],-theta[1],theta[0],]])
        A = (sym.Rational(1,2)*G)*(sym.Rational(1,2)*Gb).T                                        
        return cls(theta,A,R,G,Gb)

    @classmethod
    def EulerAnglesXYZ(cls,R,theta):
        G = sym.Matrix([  [1,0,sym.sin(theta[1])],
                                [0,sym.cos(theta[0]),-sym.sin(theta[0])*sym.cos(theta[1])],
                                [0,sym.sin(theta[1]),sym.cos(theta[0])*sym.cos(theta[1])]])
        Gb = sym.Matrix([  [sym.cos(theta[1])*sym.cos(theta[2]),sym.sin(theta[2]),0],
                                [-sym.sin(theta[2])*sym.cos(theta[1]),sym.cos(theta[2]),0],
                                [sym.sin(theta[1]),0,1]])
        A = sym.rot_axis1(-theta[0])*sym.rot_axis2(-theta[1])*sym.rot_axis3(-theta[2])
        return cls(theta,A,R,G,Gb)

    def __init__(self,theta,A,R,G,Gb):
        self._theta = sym.Matrix([*theta])
        self._G = G.copy()
        self._Gb = Gb.copy()
        super(ReferenceFrame,self).__init__(A,R)
    
    G = property(lambda self: self._G)
    Gb = property(lambda self: self._Gb)
    theta = property(lambda self: self._theta)

    def GlobalVelocity(self,u=sym.Matrix([0]*3)):
        return self.R.diff(t)-self.A*Wedge(u)*self.Gb*self.theta.diff(t)

    def BodyVelocity(self,u=sym.Matrix([0]*3)):
        return self.A.T*self.GlobalVelocity(u)

    def BodyJacobian(self,q,u=sym.Matrix([0]*3)):
        B = sym.zeros(3,len(self.theta))
        for i in range(len(self.theta)):
            B[:,i] = (self.A*u).diff(self.theta[i])
        Jb = [[self.A.T,self.A.T*B],[sym.zeros(3),self.Gb]]
        return sym.simplify(sym.BlockMatrix(Jb).as_explicit())
    
    def PuesdoSpatialFrame(self):
        return ReferenceFrame( self.theta.copy(),sym.eye(3),
                                self.t.copy(),sym.zeros(3,len(self.theta)),
                                sym.zeros(3,len(self.theta)))

    def simplify(self):
        return ReferenceFrame(self.theta.copy(),sym.simplify(self.R),
                                sym.simplify(self.t),
                                sym.simplify(self.G),
                                sym.simplify(self.Gb))

    def subs(self,*args):  
        return ReferenceFrame(self.theta.subs(*args),self.R.subs(*args),
                                self.t.subs(*args),self.G.subs(*args),
                                self.Gb.subs(*args))

    def msubs(self,*args):  
        return ReferenceFrame(me.msubs(self.theta,*args),me.msubs(self.R,*args),
                                me.msubs(self.t,*args),me.msubs(self.G,*args),
                                me.msubs(self.Gb,*args))