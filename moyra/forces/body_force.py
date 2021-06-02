import sympy as sym
from . import ExternalForce
from ..helper_funcs import linearise_matrix
import sympy.physics.mechanics as me

class BodyForce(ExternalForce):
    """A class used to represent Forces and moment in a particular reference frame"""        
    def __init__(self,p,Transform,Fx=0,Fy=0,Fz=0,Mx=0,My=0,Mz=0,simplify=True):
        """
        Constructor for a body force, with the following parameters:

        Parameters
        ----------
        p           - Instance of the ModelParameters class
        Transform   - a Homogenous transform reprewsenting the frame of refernce the force is applied in
        Fx          - (default = 0) the force applied to the body in the local x direction
        Fy          - (default = 0) the force applied to the body in the local y direction
        Fz          - (default = 0) the force applied to the body in the local z direction
        Mx          - (default = 0) the moment applied to the body about the local x axis
        My          - (default = 0) the moment applied to the body about the local y axis
        Mz          - (default = 0) the moment applied to the body about the local z axis
        """
        
        if simplify:
            BodyJacobian = sym.simplify(self._trigsimp(Transform.BodyJacobian(p.q)))
        else:
            BodyJacobian = Transform.BodyJacobian(p.q)
        wrench = sym.Matrix([Fx,Fy,Fz,Mx,My,Mz])
        super().__init__(BodyJacobian.T*wrench) 

    @staticmethod
    def _trigsimp(expr):
        return sym.trigsimp(sym.powsimp(sym.cancel(sym.expand(expr))))      




