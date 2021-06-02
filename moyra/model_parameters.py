import sympy as sym
import sympy.physics.mechanics as me

class ModelValue:
    """
    Base class to inject a value onto sympy classes
    """
    def __init__(self,value=0,**kwarg):
        super().__init__(**kwarg)
        self.value = value
        self._dependent = False      
    
    def __call__(self,x,t):
        return self._GetValue(x,t)

    def _GetValue(self,x,t):
        if callable(self.value):
            return self.value(x,t)
        else:
            return self.value

    def GetSub(self,t,x):
        return self(t,x)
        
class ModelSymbol(sym.Symbol,ModelValue):
    """
    Wrapper for Sympy Symbol, to inject it with a value attribute
    """
    def __init__(self,string,**kwarg):
        super().__init__(**kwarg)
    def __new__(cls,string,**kwarg):
        return super().__new__(cls,string)
    def __eq__(self,other):
        if isinstance(other,sym.Symbol):
            return other.name == self.name
    def __hash__(self):
        return hash(sym.Symbol(self.name))
    def _octave(self,printer):
        return f'{self.name}'

class ModelMatrixSymbol(ModelSymbol):
    def __init__(self,string,**kwarg):
        self._index = int(string.split('_')[-1])
        self._matrix = '_'.join(string.split('_')[0:-1])
        super().__init__(string,**kwarg)
    def __new__(cls,string,**kwarg):
        return super().__new__(cls,string,**kwarg)
    def _octave(self,printer):
        return f'{self._matrix}({self._index+1})'

class ModelMatrix(sym.Matrix,ModelValue):
    """
    Wrapper for Sympy Matrix, to inject it with a value attribute
    """
    def __init__(self,string,length,**kwarg):
        super().__init__(**kwarg)
        self._matrix_symbol = string
    def __new__(cls,string,length,**kwargs):
        return super().__new__(cls,sym.symbols(f'{string}_:{length}',cls=ModelMatrixSymbol))
    def __setattr__(self,name,value):
        if name == "value":
            if value is not None:
                r, c = self.shape
                if len(value) != r*c:
                    raise ValueError(f'Model Matrix value length, {len(value)}, must be the same length as the symbolic matrix, {self.shape}.')
        object.__setattr__(self, name, value)



class ModelExpr(sym.Symbol,ModelValue):
    def __init__(self,string,func,**kwarg):
        self.expr_func = func
        super().__init__(**kwarg)

    def _GetValue(self,t,x):
        return self.expr_func(t,x)

    def __new__(cls,string,**kwarg):
        return super().__new__(cls,string)

    def GetSub(self,t,x):
        return self.value
class ModelParameters:    
    
    def GetTuple(self,ignore=[]):
        return tuple(var for name,var in vars(self).items() if isinstance(var,ModelValue) and name not in ignore and var not in ignore)
    
    def GetSubs(self,t,x,ignore=[]):
        sub_dependent_dict = {}
        sub_dict = {}
        # put dependent substitions in first
        for name,var in vars(self).items():
            if isinstance(var,ModelValue) and name not in ignore and var not in ignore:
                if isinstance(var,ModelMatrix):
                    for i in range(len(var)):
                        sub_dict[var[i]] = var.GetSub(t,x)[i]
                else:
                    if var._dependent:
                        sub_dependent_dict[sym.Symbol(var.name)] = var.GetSub(t,x)
                    else:
                        sub_dict[sym.Symbol(var.name)] = var.GetSub(t,x)
        # sub in values for dependent subsitutions
        for key,value in sub_dependent_dict.items():
            sub_dependent_dict[key] = value.subs(sub_dict)
        #combine dictionaries
        tot_sub_dict = {**sub_dict,**sub_dependent_dict}
        # return a dictionary with all keys changed to symbols
        return tot_sub_dict#{sym.Symbol(k.name):v for k,v in tot_sub_dict.items()}
    
    def GetNumericTuple(self,x,t,ignore=[]):
        return tuple(var(x,t) for name,var in vars(self).items() if isinstance(var,ModelValue) and name not in ignore and var not in ignore)

    def to_matlab_class(self,class_name = "Parameters",file_dir='', ignore=[]):
        import os.path
        # create a dict of all required params and values
        params = {}
        for name,var in vars(self).items():
            if name not in ignore and var not in ignore:
                if isinstance(var,ModelSymbol):
                    params[var.name] = var.value
                if isinstance(var,ModelMatrix):
                    params[name] = var.value
        # convert to matlab class string
        classdef = classdef = f'classdef {class_name}'
        params = '\n\t\t'.join([ f'{key} = {value}' for key,value in params.items()])
        class_string = classdef + '\n\tproperties\n\t\t' + params + '\n\tend\nend'
        # save to file
        with open(os.path.join(file_dir,class_name + '.m'),'w') as file:
            file.write(class_string)