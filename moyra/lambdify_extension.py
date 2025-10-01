# from sympy.utilities.lambdify import _EvaluatorPrinter
# from sympy.core.compatibility import (is_sequence, iterable,
#     NotIterable)
# import builtins

from sympy.utilities.lambdify import _EvaluatorPrinter
from sympy.utilities.iterables import is_sequence, iterable
from sympy.core.compatibility import NotIterable
import builtins

def doprint(self, funcname, args, expr):
        """Returns the function definition code as a string."""
        # Created by lambdify in sympy.utilities.lambdify
        from sympy.utilities.lambdify import StringIO, _is_safe_int

        func_body = []

        # Generate argument list
        arg_list = []
        for arg in args:
            if iterable(arg):
                arg_list.append(self._print(arg))
            else:
                arg_list.append(self.dummy_name(arg))
        arg_list = ", ".join(arg_list)

        # Generate function body
        if self.cse:
            sub_expressions, simplified_expr = self._cse(expr)
            for var, sub_expr in sub_expressions:
                func_body.append("  %s = %s" % (var, self._print(sub_expr)))
            expr = simplified_expr[0]

        if self.use_imps:
            func_body.append("  return (%s)" % self._print(expr))
        else:
            result = self._print(expr)
            if result.startswith('(') and result.endswith(')'):
                func_body.append("  return %s" % result)
            else:
                func_body.append("  return (%s)" % result)

        # Generate function definition
        func_head = "def %s(%s):" % (funcname, arg_list)

        return "\n".join([func_head] + func_body)
