import sympy as sym
from .base_element import BaseElement

class Damper(BaseElement):
    def __init__(self,velocity,damping_constant):
        self.__c = damping_constant
        self.__z_dot = velocity
    def calc_ke(self,p):
        return 0
    def calc_pe(self,p):
        return 0
    def calc_rdf(self,p):
        return sym.Rational(1,2)*self.__c*self.__z_dot**2