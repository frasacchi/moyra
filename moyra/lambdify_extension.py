# from sympy.utilities.lambdify import _EvaluatorPrinter
# from sympy.utilities.iterables import is_sequence, iterable
# from sympy.core.compatibility import NotIterable
# import builtins

from sympy.utilities.lambdify import _EvaluatorPrinter
from sympy.core.compatibility import (exec_, is_sequence, iterable,
    NotIterable, builtins)
    
def doprint(self, funcname, args, expr):
        """Returns the function definition code as a string."""
        from sympy import Dummy,cse,Symbol

        funcbody = []

        if not iterable(args):
            args = [args]

        argstrs, expr = self._preprocess(args, expr)

        ## --------------- Addition -----------------
        replacments, exprs = cse(expr,symbols=(Symbol(f'rep_{i}')for i in range(10000)))
        if isinstance(expr,tuple):
            expr = tuple(exprs)
        elif isinstance(expr,list):
            expr = exprs
        else:
            result = self._print(expr)
            if result.startswith('(') and result.endswith(')'):
                func_body.append("  return %s" % result)
            else:
                func_body.append("  return (%s)" % result)

        # Generate function definition
        func_head = "def %s(%s):" % (funcname, arg_list)

        return "\n".join([func_head] + func_body)
