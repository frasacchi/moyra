from sympy.printing.python import PythonPrinter
import keyword as kw
class ModelPrinter(PythonPrinter):
    def __init__(self, settings=None):
        super(ModelPrinter, self).__init__(settings)
        self.model_symbols = []
    
    def _print_ModelMatrixSymbol(self,expr):
        symbol = self._str(expr)
        index = int(expr.name.split('_')[-1])
        matrix = '_'.join(expr.name.split('_')[0:-1])
        if symbol not in self.model_symbols:
            self.model_symbols.append(symbol)
        return f'p.{matrix}[{index}]'
    
    def _print_ModelSymbol(self,expr):
        symbol = self._str(expr)
        if symbol not in self.model_symbols:
            self.model_symbols.append(symbol)
        return f'p.{expr.name}'

def model(expr,p,**settings):
    """Return Python interpretation of passed expression
    (can be passed to the exec() function without any modifications)"""
    printer = ModelPrinter(settings)
    exprp = printer.doprint(expr)

    result = ''
    # Returning found symbols and functions
    renamings = {}
    for symbolname in printer.symbols:
        newsymbolname = symbolname
        # Escape symbol names that are reserved python keywords
        if kw.iskeyword(newsymbolname):
            while True:
                newsymbolname += "_"
                if (newsymbolname not in printer.symbols and
                        newsymbolname not in printer.functions):
                    renamings[sympy.Symbol(
                        symbolname)] = sympy.Symbol(newsymbolname)
                    break
        result += newsymbolname + ' = Symbol(\'' + symbolname + '\')\n'

    for functionname in printer.functions:
        if functionname in [i.name for i in p.q]:
            continue
        newfunctionname = functionname
        # Escape function names that are reserved python keywords
        if kw.iskeyword(newfunctionname):
            while True:
                newfunctionname += "_"
                if (newfunctionname not in printer.symbols and
                        newfunctionname not in printer.functions):
                    renamings[sympy.Function(
                        functionname)] = sympy.Function(newfunctionname)
                    break
        result += newfunctionname + ' = Function(\'' + functionname + '\')\n'

    if renamings:
        exprp = expr.subs(renamings)
    exprp_str = printer._str(exprp)
    for i,q_i in enumerate(p.q):
        exprp_str = exprp_str.replace(f'Derivative({q_i.name}(t), t)',f'p.qd[{i}]').replace(f'{q_i.name}(t)',f'p.q[{i}]')
    result += 'e = ' + exprp_str
    return result