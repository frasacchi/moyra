import sympy as sym
import sympy.physics.mechanics as me
import os
from .model_parameters import ModelParameters,ModelSymbol,ModelMatrix

class DynamicModelParameters(ModelParameters):
    def __init__(self,DoFs):
        self.qs = DoFs
        self.q = sym.Matrix(me.dynamicsymbols(f'q:{DoFs}'))
        self.qd = sym.Matrix(me.dynamicsymbols(f'q:{DoFs}',1))
        self.qdd = sym.Matrix(me.dynamicsymbols(f'q:{DoFs}',2))

        # create state matrix
        x_ls = []
        for i in range(0,DoFs):
            x_ls.append(self.q[i])
            x_ls.append(self.qd[i])
        self.x = sym.Matrix(x_ls)
        super().__init__()

    def print_python(self):
        ignore = ['qs','q','qd','qdd']
        string = f'p = ma.DynamicModelParameters({self.qs})'
        for name,value in vars(self).items():
            if name not in ignore:
                if isinstance(value, ModelSymbol):
                    if callable(value.value):
                        string += f'\np.{name} = ma.ModelSymbol(string=\'{value.name}\')'
                    else:
                        string += f'\np.{name} = ma.ModelSymbol(value={value.value}, string=\'{value.name}\')'
                elif isinstance(value, ModelMatrix):
                    if callable(value.value):
                        string += f'\np.{name} = ma.ModelMatrix(string=\'{value._matrix_symbol}\', length={len(value)})'
                    else:
                        string += f'\np.{name} = ma.ModelMatrix(value={value.value}, string=\'{value._matrix_symbol}\', length={len(value)})'
                elif isinstance(value,sym.Symbol):
                    string += f'\np.{name} = Symbol(\'{value.name}\')'
        return string

    def to_file(self,filename,file_dir=''):
        string = 'import moyra as ma\nfrom sympy import *\n\ndef get_p():\n\t' + self.print_python().replace('\n','\n\t') + '\n\treturn p\n'
        with open(os.path.join(file_dir,filename),'w') as file:
            file.write(string)