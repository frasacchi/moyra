from sympy.abc import x,beta
from moyra.elements import *
from moyra import ModelParameters,ModelSymbol
import pytest


@pytest.fixture
def p():
    p = ModelParameters()
    p.x = ModelSymbol(value = 1, string = 'x')
    p.beta = ModelSymbol(value = 1, string = 'beta')
    return p

@pytest.fixture
def spring_ele(p):
    return Spring(p.x,p.beta)

def test_spring_ele(spring_ele,p):
    assert spring_ele.calc_pe(p) == sym.Rational(1,2)*beta*x**2

def test_spring_ele(spring_ele,p):
    assert spring_ele.calc_ke(p) == 0

def test_spring_ele(spring_ele,p):
    assert spring_ele.calc_rdf(p) == 0