import sympy as sym
import sympy.physics.mechanics as me

class ModelValue:
    """
    Base class to inject a value onto sympy classes
    """
    def __init__(self,value=0,comment=None,**kwarg):
        # super().__init__(**kwarg)
        self.value = value
        self.comment = comment
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

class OctaveZero(sym.Symbol):
    """
    Wrapper for Sympy Symbol, to inject it with a value attribute
    """
    def _octave(self,printer):
        return f'0'
    def _cxxcode(self,printer):
        return f'0'      
class VarElement(sym.matrices.expressions.matexpr.MatrixElement):
    def _octave(self,printer):
        return f'{self.parent.name}({self.i+1},:)'
class VarVector(sym.MatrixSymbol):
    def __new__(cls,string,i,j):
        return super().__new__(cls,string,i,j)
    def _entry(self, i, j, **kwargs):
        return VarElement(self, i, j)
    def _octave(self,printer):
        return f'{self.name}'
    
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
    def _cxxcode(self,printer):
        return f'{self.name}'

class ModelVectorSymbol(ModelSymbol):
    def __init__(self,string,**kwarg):
        self._index = int(string.split('_')[-1])
        self._matrix = '_'.join(string.split('_')[0:-1])
        super().__init__(string,**kwarg)
    def __new__(cls,string,**kwarg):
        return super().__new__(cls,string,**kwarg)
    def _octave(self,printer):
        return f'{self._matrix}({self._index+1})'
    def _cxxcode(self,printer):
        return f'{self._matrix}({self._index},0)'

class ModelVector(sym.Matrix,ModelValue):
    """
    Wrapper for Sympy Vector, to inject it with a value attribute
    """
    def __init__(self,string,length,**kwarg):
        if "value" not in kwarg:
            kwarg["value"] = [0]*length
        super().__init__(**kwarg)
        self._matrix_symbol = string
    def __new__(cls,string,length,**kwargs):
        return super().__new__(cls,sym.symbols(f'{string}_:{length}',cls=ModelVectorSymbol))
    def __setattr__(self,name,value):
        if name == "value":
            if value is not None:
                r, c = self.shape
                if len(value) != r*c:
                    raise ValueError(f'Model Vector value length, {len(value)}, must be the same length as the symbolic matrix, {self.shape}.')
        object.__setattr__(self, name, value)
        

class ModelMatrixSymbol(ModelSymbol):
    def __init__(self,string,**kwarg):
        self._index = [int(i) for i in string.split('_')[-1].split(',')]
        self._matrix = '_'.join(string.split('_')[0:-1])
        super().__init__(string,**kwarg)
    def __new__(cls,string,**kwarg):
        return super().__new__(cls,string,**kwarg)
    def _octave(self,printer):
        return f'{self._matrix}({self._index[0]+1},{self._index[1]+1})'
    def _cxxcode(self,printer):
        return f'{self._matrix}({self._index[0]},{self._index[1]})'       
class ModelMatrix(sym.Matrix,ModelValue):
    """
    Wrapper for Sympy Matrix, to inject it with a value attribute
    """
    def __init__(self,string,size,**kwarg):
        if "value" not in kwarg:
            kwarg["value"] = [[0]*size[1]]*size[0]
        super().__init__(**kwarg)
        self._matrix_symbol = string
    def __new__(cls,string,size,**kwargs):
        syms = [[ModelMatrixSymbol(f'{string}_{j},{i}') for i in range(size[1])] for j in range(size[0])]
        return super().__new__(cls,syms)
    def __setattr__(self,name,value):
        if name == "value":
            if value is not None:
                r, c = self.shape
                rv,cv = len(value),len(value[0])
                if r != rv and c != cv:
                    raise ValueError(f'Model Matrix value Shape must be the same shape as the symbolic matrix, {self.shape}.')
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

    def to_matlab_class(self,class_name = "Parameters",file_dir='', ignore=[], base_class = None):
        import os.path
        # create a dict of all required params and values
        params = {}
        for name,var in vars(self).items():
            if name not in ignore and var not in ignore:
                if isinstance(var,ModelSymbol):
                    params[var.name] = [var.value,var.comment]
                if isinstance(var,ModelVector):
                    params[name] = [var.value,var.comment]
                if isinstance(var,ModelMatrix):
                    params[name] = ['['+';'.join([f'{v}' for v in var.value])+']',var.comment]
        # convert to matlab class string
        cn = class_name if base_class is None else class_name+f" < {base_class}"
        classdef = f'classdef {cn}'
        params = '\n\t\t'.join([ f'{key} = {value[0]}' + ('' if value[1] is None else f'\t\t% {value[1]}') for key,value in params.items()]).replace('{','').replace('}','')
        class_string = classdef + '\n\tproperties\n\t\t' + params + '\n\tend\nend'
        # save to file
        with open(os.path.join(file_dir,class_name + '.m'),'w') as file:
            file.write(class_string)

    def to_cxx_class(self,class_name = "Parameters",file_dir='', ignore=[],func_sigs = [], base_class = None, additional_includes = [],header_ext = 'hpp',func_pragma = None):
        import os.path
        # create a dict of all required params and values
        cn = class_name if base_class is None else class_name+f" : public {base_class} "
        classdef = f'class {cn}{{'

        params = []
        initilisers = []
        for name,var in vars(self).items():
            if name not in ignore and var not in ignore:
                if isinstance(var,ModelSymbol):
                    params.append(f'double {var.name} = {var.value};')
                if isinstance(var,ModelMatrix):
                    type_str = 'VectorXd'
                    params.append(f'{type_str} {name} = {type_str}({len(var.value)});')
                    initilisers.append(f'{name} << '+', '.join((str(i) for i in var.value))+';')
        # convert to matlab class string
        pragma = '#pragma once\n'
        includes = ['"Eigen/Core"','"Eigen/Dense"',*additional_includes]
        typedefs = ['Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXd','Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXd','Eigen::Matrix<double, 3, 1> Vector3d'];
        includes = ''.join(('#include '+i+'\n' for i in includes))
        typedefs = ''.join(('typedef '+i+';\n' for i in typedefs))
        params = '\n\t'.join(params).replace('{','').replace('}','')
        funcs_string = ';\n\t'.join(func_sigs).replace('{','').replace('}','')+';'
        initisler = f'{class_name}(){{\n\t\t' + '\n\t\t'.join(initilisers) + '\n\t};'
        initisler = initisler if func_pragma is None else f'{func_pragma} {initisler}'
        destructor = f'~{class_name}()'+'{};\n\t'
        destructor = destructor if func_pragma is None else f'{func_pragma} {destructor}'
        class_string = pragma + includes + typedefs + classdef + '\n\tpublic:\n\t' + params + '\n\t' + funcs_string +'\n\t'+ initisler + '\n\t' + destructor +'\n};'
        # save to file
        with open(os.path.join(file_dir,f'{class_name}.{header_ext}'),'w') as file:
            file.write(class_string)