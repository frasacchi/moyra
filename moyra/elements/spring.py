import sympy as sym
from .base_element import BaseElement

class Spring(BaseElement):
    def __init__(self,deflection,spring_constant):
        self.__k = spring_constant
        self.__z = deflection
    def calc_ke(self,p):
        return 0
    def calc_pe(self,p):
        return sym.Rational(1,2)*self.__k*self.__z**2
    def calc_rdf(self,p):
        return 0
        
