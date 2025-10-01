import re
import os
import pickle
import sympy as sym
import numpy as np
from scipy.linalg import eig
import sympy.physics.mechanics as me
from sympy.physics.vector.printing import vpprint, vlatex
from sympy.abc import x,y,t

from moyra.forces.external_force import ExternalForce
from .helper_funcs import linearise_matrix, partial_wrt_t
from .numeric_model import NumericModel
import pickle
from .printing import model as print_model
from .model_parameters import ModelSymbol, ModelMatrix,ModelMatrixSymbol, ModelValue, ModelVectorSymbol, ModelVector, VarVector, VarElement, OctaveZero
from time import time, ctime
from collections.abc import Iterable
from sympy.abc import t
from .lambdify_extension import SymbolInterator

class SymbolicModel:
    """
    An instance of a folding wing tip model using assumed shapes.

    Required inputs are:
        generalisedCoords - array of the generalised coordinate symbols 
            (must be dynamic symbols)
        z_w,z_t,alpha_w,alpha_t - sympy expressions of the z and alpha postion
            of the wing and FWT
        FwtParameters - instance of the FwtParameters class (with the symbols 
            used in the above expressions)
        thetaIndex - index of theta (hinge angle) in generalisedCoords 
            (so energy equation knows which one) if no theta coordinate leave
            as 'None'
    """
    @classmethod
    def FromElementsAndForces(cls,q,Elements, ExtForces=None, C=None, legacy=False):
        """
        Create a symbolic Model instance from a set Elements and external forces
        """
        # Calc K.E, P.E and Rayleigh Dissaptive Function
        T = U = D = sym.Integer(0)
        qs = len(q)
        M = sym.zeros(qs)
        f = sym.zeros(qs,1)
        sm = cls(q,M,f,T,U)
        # add K.E for each Rigid Element
        Elements = Elements if isinstance(Elements,Iterable) else [Elements]
        for i,ele in enumerate(Elements):
            print(f'Generating EoM for Element {i+1} out of {len(Elements)} - {ele}')
            sm += ele.to_symbolic_model(legacy=legacy)
        
        return cls(q,sm.M,sm.f,sm.T,sm.U,ExtForces,C)

    def __init__(self,q,M,f,T,U,ExtForces = None,C = sym.Matrix([])):
        """Initialise a Symbolic model of the form 
        $M\\ddot{q}+f(\\dot{q},q,t)-ExtForces(\\dot{q},q,t) = 0, with constraints C=0$

        with the Symbolic Matricies M,f,and Extforces
        """
        self._M = M
        self._f = f
        self._T = T
        self._U = U
        self._q = q
        self._ExtForces = ExtForces if ExtForces is not None else ExternalForce.zero_force(f.shape[0])
        self._C = C

    M = property(lambda self:self._M)
    f = property(lambda self:self._f)
    T = property(lambda self:self._T)
    U = property(lambda self:self._U)
    q = property(lambda self:self._q)
    qd = property(lambda self:self.q.diff(t))
    qdd = property(lambda self:self.q.diff(t,2))
    qs = property(lambda self:len(self.q))
    ExtForces = property(lambda self:self._ExtForces)
    C = property(lambda self:self._C)

    def __add__(self,other):
        if not isinstance(other,SymbolicModel):
            raise TypeError('other must be of type SymbolicModel')
        idx = [i for i, e in enumerate(self.q) if e in other.q]
        
        # add mass matricies
        M = sym.Matrix(self.M)
        for i,m_i in enumerate(idx):
            for j,m_j in enumerate(idx):
                M[m_i,m_j] += other.M[i,j]

        # add forces matricies
        f = sym.Matrix(self.f)
        Q = self.ExtForces.Q()
        Q_other = other.ExtForces.Q()
        for i,m_i in enumerate(idx):
                f[m_i] += other.f[i]
                Q[m_i] += Q_other[i]

        T = self.T + other.T
        U = self.U + other.U
        C = sym.Matrix([*self.C,*other.C])

        return SymbolicModel(self.q,M,f,T,U,ExternalForce(Q),C)

    def simplify(self):
        """
        Creates a new instance of a Symbolic model with the simplify simplifcation applied
        """
        ExtForces = self.ExtForces.simplify() if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else sym.simplify(self.T)
        U = self.U if isinstance(self.U,int) else sym.simplify(self.U)
        C = self.C if self.C is None else sym.simplify(self.C)
        return SymbolicModel(self.q,sym.simplify(self.M),sym.simplify(self.f),
                            T,U,ExtForces,C)        

    def cancel(self):
        """
        Creates a new instance of a Symbolic model with the cancel simplifcation applied
        """
        ExtForces = self.ExtForces.cancel() if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else sym.cancel(self.T)
        U = self.U if isinstance(self.U,int) else sym.cancel(self.U)
        C = self.C if self.C is None else sym.cancel(self.C)
        return SymbolicModel(self.q,sym.cancel(self.M),sym.cancel(self.f),
                            T,U,ExtForces,C)

    def expand(self):
        """
        Creates a new instance of a Symbolic model with the cancel simplifcation applied
        """
        ExtForces = self.ExtForces.expand() if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else sym.expand(self.T)
        U = self.U if isinstance(self.U,int) else sym.expand(self.U)
        C = self.C if self.C is None else sym.expand(self.C)
        return SymbolicModel(self.q,sym.expand(self.M),sym.expand(self.f),
                            T,U,ExtForces,C)


    def subs(self,*args):
        """
        Creates a new instance of a Symbolic model with the substutions supplied
         in args applied to all the Matricies
        """
        ExtForces = self.ExtForces.subs(*args) if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else self.T.subs(*args)
        U = self.U if isinstance(self.U,int) else self.U.subs(*args)
        C = self.C if self.C is None else self.C.subs(*args)
        return SymbolicModel(self.q,self.M.subs(*args),self.f.subs(*args),
                            T,U,ExtForces,C)

    def msubs(self,*args):
        """
        Creates a new instance of a Symbolic model with the substutions supplied
         in args applied to all the Matricies
        """
        ExtForces = self.ExtForces.msubs(*args) if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else me.msubs(self.T,*args)
        U = self.U if isinstance(self.U,int) else me.msubs(self.U,*args)
        C = self.C if self.C is None else me.msubs(self.C,*args)
        return SymbolicModel(self.q, me.msubs(self.M,*args),me.msubs(self.f,*args),
                            T,U,ExtForces,C)

    def linearise(self,x_f):
        """
        Creates a new instance of the symbolic model class in which the EoM have been 
        linearised about the fixed point p.q_0
        """
        # Calculate Matrices at the fixed point
        # (go in reverse order so velocitys are subbed in before positon)

        # get the full EoM's for free vibration and linearise

        x = [*self.q,*self.qd]
        x_subs = {x[i]:x_f[i] for i in range(len(x))}

        M_lin = me.msubs(self.M,x_subs)

        f_lin = linearise_matrix(self.f,x,x_f)
        T_lin = linearise_matrix(self.T,x,x_f)
        U_lin = linearise_matrix(self.U,x,x_f)

        # Linearise the External Forces
        extForce_lin = self.ExtForces.linearise(x,x_f) if self.ExtForces is not None else None

        # create the linearised model and return it
        return SymbolicModel(self.q,M_lin,f_lin,T_lin,U_lin,extForce_lin)

    def extract_matrices(self):
        """
        From the current symbolic model extacts the classic matrices A,B,C,D,E as per the equation below
        A \\ddot{q} + B\\dot{q} + Cq = D\\dot{q} + Eq

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        A = self.M
        D = self.f.jacobian(self.qd)
        E = self.f.jacobian(self.q)
        B = self.ExtForces.Q().jacobian(self.qd)
        C = self.ExtForces.Q().jacobian(self.q)
        return A,B,C,D,E

    def free_body_eigen_problem(self):
        """
        gets the genralised eigan matrices for the free body problem.
        They are of the form:
            |   I   0   |       |    0    I   |
        M=  |   0   M   |   ,K= |   -C   -B   |
        such that scipy.linalg.eig(K,M) solves the problem 

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        M = sym.eye(self.qs*2)
        M[-self.qs:,-self.qs:]=self.M

        K = sym.zeros(self.qs*2)
        K[:self.qs,-self.qs:] = sym.eye(self.qs)
        K[-self.qs:,:self.qs] = -self.f.jacobian(self.q)
        K[-self.qs:,-self.qs:] = -self.f.jacobian(self.qd)
        return K,M

    def gen_eigen_problem(self):
        """
        gets the genralised eigan matrices for use in solving the frequencies / modes. 
        They are of the form:
            |   I   0   |       |    0    I   |
        M=  |   0   M   |   ,K= |   E-C  D-B  |
        such that scipy.linalg.eig(K,M) solves the problem 

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        M_prime = sym.eye(self.qs*2)
        M_prime[-self.qs:,-self.qs:]=self.M

        _Q = self.ExtForces.Q() if self.ExtForces is not None else sym.Matrix([0]*self.qs)

        K_prime = sym.zeros(self.qs*2)
        K_prime[:self.qs,-self.qs:] = sym.eye(self.qs)
        K_prime[-self.qs:,:self.qs] = _Q.jacobian(self.q)-self.f.jacobian(self.q)
        K_prime[-self.qs:,-self.qs:] = _Q.jacobian(self.qd)-self.f.jacobian(self.qd)

        return K_prime, M_prime

    @staticmethod
    def _jacobian(M,x):
        return sym.Matrix([[*M.diff(xi)] for xi in x]).T

    def to_file(self,p,filename):
        #Get string represtations
        M_code = "def get_M(p):\n\t"+print_model(self.M,p).replace('\n','\n\t')+"\n\treturn e\n"
        f_code = "def get_f(p):\n\t"+print_model(self.f,p).replace('\n','\n\t')+"\n\treturn e\n"
        T_code = "def get_T(p):\n\t"+print_model(self.T,p).replace('\n','\n\t')+"\n\treturn e\n"
        U_code = "def get_U(p):\n\t"+print_model(self.U,p).replace('\n','\n\t')+"\n\treturn e\n"
        p_code = 'def get_p():\n\t'+p.print_python().replace('\n','\n\t')+"\n\treturn p\n"

        if self.ExtForces is not None:
            Q_code = "def get_Q(p):\n\t"+print_model(self.ExtForces.Q(),p).replace('\n','\n\t')+"\n\treturn e\n"
        else:
            Q_code = "def get_Q(p):\n\t"+"return ImmutableDenseMatrix([[0]"+",[0]"*(self.M.shape[0]-1)+"])\n"
        if self.C is not None:
            C_code = 'def get_C(p):\n\t'+print_model(self.C,p).replace('\n','\n\t')+"\n\treturn e\n"
        else:
            C_code = 'def get_C(p):\n\treturn None\n'
        #Combine and add import statements
        full_code = "from sympy import *\nimport moyra as ma\n\n"+M_code+f_code+T_code+U_code+Q_code+C_code+p_code

        # Save to the file
        t_file = open(filename,"w")
        n = t_file.write(full_code)
        t_file.close()   


    @classmethod
    def from_file(cls,filename):
        import importlib.util
        from .forces import ExternalForce
        spec = importlib.util.spec_from_file_location("my.Model", filename)
        m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(m)
        p = m.get_p()
        M = m.get_M(p)
        f = m.get_f(p)
        T = m.get_T(p)
        U = m.get_U(p)
        _Q = m.get_Q(p)
        C = m.get_C(p)
        ExtForce = ExternalForce(_Q)
        return (cls(M,f,T,U,ExtForce,C),p)


    def to_matlab_class(self,p,file_dir,class_name,base_class = None,additional_funcs = [], octave_user_functions={}):
        funcs = [('get_M',self.M),('get_f',self.f),('get_Q',self.ExtForces.Q()),
                ('get_KE',self.T),('get_PE',self.U)]
        funcs = [*funcs,*additional_funcs]
        if self.C is not None:
            funcs.append(('get_C',self.C))
            C_q = sym.simplify(self.C).jacobian(self.q)
            C_t =  partial_wrt_t(self.C,p.q)
            C_tt = partial_wrt_t(C_t,p.q)
            C_qt = partial_wrt_t(C_q,p.q)
            Q_c = C_tt + (C_q*p.qd).jacobian(p.q)*p.qd + 2*C_qt*p.qd
            M_lag = sym.BlockMatrix([[self.M,C_q.T],[C_q,sym.zeros(len(self.C))]]).as_explicit()
            Q_lag = sym.BlockMatrix([[self.f-self.ExtForces.Q()],[Q_c]]).as_explicit()

            funcs.append(('get_C_q',C_q))
            funcs.append(('get_C_t',C_t))
            # funcs.append(('get_C_tt',me.msubs(C_tt))
            funcs.append(('get_Q_c',Q_c))
            funcs.append(('get_M_lag',M_lag))
            funcs.append(('get_Q_lag',Q_lag))
        # create directory
        class_dir = os.path.join(file_dir,f"@{class_name}")
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for func_name,matrix in funcs:
            with open(os.path.join(class_dir,f"{func_name}.m"),'w') as file:
                file.write(self._gen_octave(matrix,func_name,octave_user_functions))
        p.to_matlab_class(class_name=class_name, file_dir=class_dir, base_class=base_class )
    
    def to_cxx_class(self,p,file_dir,class_name,base_class = None,additional_funcs = [], additional_includes = [],header_ext='hpp',source_ext = 'cpp',func_pragma=None):
        funcs = [('get_M',self.M,'MatrixXd'),('get_f',self.f),('get_Q',self.ExtForces.Q()),
                ('get_KE',self.T),('get_PE',self.U)]
        funcs = [*funcs,*additional_funcs]
        if self.C is not None:
            funcs.append(('get_C',self.C))
            C_q = sym.simplify(self.C).jacobian(self.q)
            C_t =  partial_wrt_t(self.C,p.q)
            C_tt = partial_wrt_t(C_t,p.q)
            C_qt = partial_wrt_t(C_q,p.q)
            Q_c = C_tt + (C_q*p.qd).jacobian(p.q)*p.qd + 2*C_qt*p.qd
            M_lag = sym.BlockMatrix([[self.M,C_q.T],[C_q,sym.zeros(len(self.C))]]).as_explicit()
            Q_lag = sym.BlockMatrix([[self.f-self.ExtForces.Q()],[Q_c]]).as_explicit()

            funcs.append(('get_C_q',C_q))
            funcs.append(('get_C_t',C_t))
            # funcs.append(('get_C_tt',me.msubs(C_tt))
            funcs.append(('get_Q_c',Q_c))
            funcs.append(('get_M_lag',M_lag))
            funcs.append(('get_Q_lag',Q_lag))
        # create directory
        class_dir = os.path.join(file_dir,f"{class_name}")
        func_sigs = []
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        for val in funcs:
            if len(val)==3:
                func_name,matrix,out_type = val
            else:
                func_name,matrix = val
                out_type = None
            with open(os.path.join(class_dir,f"{func_name}.{source_ext}"),'w') as file:
                (cxx_str,cxx_sig) = self._gen_cxx(matrix,func_name,class_name,out_type=out_type,header_ext=header_ext,func_pragma=func_pragma)
                file.write(cxx_str)
                func_sigs.append(cxx_sig)
        p.to_cxx_class(class_name=class_name, file_dir=class_dir, base_class=base_class, func_sigs=func_sigs, additional_includes=additional_includes,header_ext = header_ext,func_pragma=func_pragma)

    def to_matlab_file(self,p,file_dir,octave_user_functions={}):
        funcs = (('get_M',self.M),('get_f',self.f),('get_Q',self.ExtForces.Q()))
        for func_name,matrix in funcs:
            with open(os.path.join(file_dir,f"{func_name}.m"),'w') as file:
                file.write(self._gen_octave(matrix,func_name,octave_user_functions))
        p.to_matlab_class(file_dir=file_dir)

    def to_matlab_file_linear(self,p,file_dir,octave_user_functions={}):
        mats = self.extract_matrices()
        names = ['A','B','C','D','E']
        funcs = list(zip([f'get_{i}' for i in names],mats))
        for func_name,matrix in funcs:
            with open(os.path.join(file_dir+f"{func_name}.m"),'w') as file:
                file.write(self._gen_octave(matrix,func_name,octave_user_functions))
        p.to_matlab_class(file_dir=file_dir)
        
    def _gen_cxx(self,expr,func_name,class_name=None,out_type=None,header_ext='hpp',func_pragma = None):
        # convert states to non-time dependent variable
        U = ModelVector(string='U',length=int(self.qs*2))
        l = dict(zip([*self.q,*self.qd],U))
        l_deriv = dict(zip(self.q.diff(t),self.qd))
        expr = me.msubs(expr,l_deriv)
        expr = me.msubs(expr,l)

        # get parameter replacements
        param_string = '// extract required parameters from structure\n\t'
        unknown_vars = []
        for var in expr.free_symbols:
            if isinstance(var,ModelValue):
                pass
            elif var not in U:
                print(f'Unknown variable {var} found in function {func_name}. It will be added to the function signature.')
                if isinstance(var,sym.matrices.expressions.matexpr.MatrixSymbol):
                    unknown_vars.append((var,'VectorXd'))
                elif isinstance(var,sym.matrices.dense.MutableDenseMatrix):
                    unknown_vars.append((var,'VectorXd'))
                else:
                    unknown_vars.append((var,'double'))


        # split expr into groups
        replacments, exprs = sym.cse(expr,symbols=SymbolInterator())
        if isinstance(expr,tuple):
            expr = tuple(exprs)
        elif isinstance(expr,list):
            expr = exprs
        else:
            expr = exprs[0]      

        group_string = '// create common groups\n\t'
        for variable, expression in replacments:
            group_string +=f'double {variable} = {sym.printing.cxxcode(expression)};\n\t'
        
        # convert to octave string and covert states to vector form
        out = '// create output vector\n\t'
        
        if isinstance(expr,sym.matrices.dense.MutableDenseMatrix):
            numel = 1
            shape = expr.shape
            for i in range(len(shape)):
                numel *= shape[i]
            if shape[1] == 1 and out_type != 'MatrixXd':
                out_type = f'VectorXd'
                out += f'{out_type} out = {out_type}({shape[0]});\n\t'
            else:
                out_type = f'MatrixXd'
                out += f'{out_type} out = {out_type}({shape[0]},{shape[1]});\n\t'                    
            for j in range(shape[0]):
                for k in range(shape[1]):
                    out += f'\n\tout({j},{k}) = ' + sym.printing.cxxcode(expr[j,k]) + ';'
        else:
            out += 'double out = ' + sym.printing.cxxcode(expr) + ';'
            out_type = 'double'
        out += '\n\treturn out;'

        file_sig = f'// {func_name.upper()} Auto-generated function from moyra\n\t'
        file_sig += f'// \n\t'
        file_sig += f'// \tCreated at : {ctime(time())} \n\t'
        file_sig += f'// \tCreated with : moyra https://pypi.org/project/moyra/\n\t'
        file_sig += f'// \n\t'

        # wrap output in octave function signature
        # create unknow var string
        unknown_str = '' if not unknown_vars else ','+','.join(sorted([ty + ' &' + str(i) for (i,ty) in unknown_vars], key=str))
        class_func_name = func_name if class_name is None else f'{class_name}::{func_name}'
        signature = f'{out_type} {class_func_name}(VectorXd &U{unknown_str}){{\n\t'
        signature = signature if func_pragma is None else f'{func_pragma} {signature}'
        includes = []
        if class_name is not None:
            includes.append(f'{class_name}.{header_ext}')
        if not includes:
            includes = ''
        else:
            includes = ''.join(('#include "'+i+'"\n' for i in includes))
        
        ## tidy params with curly braces in name
        # my_replace = lambda x: f'_{x.group(1)}'
        # param_string = re.sub(r"_\{(?P<index>.+)\}",my_replace,param_string)
        # group_string = re.sub(r"_\{(?P<index>.+)\}",my_replace,group_string)
        # out = re.sub(r"_\{(?P<index>.+)\}",my_replace,out)

        cxx_string = includes + '\n' + signature + file_sig + param_string + group_string + out + '\n};'
        
        signature = f'{out_type} {func_name}(VectorXd &U{unknown_str})'
        signature = signature if func_pragma is None else f'{func_pragma} {signature}'
        return (cxx_string,signature)

        
    def _gen_octave(self,expr,func_name, user_functions={}):
        # convert states to non-time dependent variable
        U = sym.Matrix(sym.symbols(f'u_:{self.qs*2}'))
        l = dict(zip([*self.q,*self.qd],U))
        l_deriv = dict(zip(self.q.diff(t),self.qd))
        expr = me.msubs(expr,l_deriv)
        expr = me.msubs(expr,l)

        # get parameter replacements
        param_string = '%% extract required parameters from structure\n\t'
        matries = []
        unknown_vars = []
        varVector = []
        isVarVector = False
        for var in expr.free_symbols:
            if isinstance(var,ModelValue):
                if isinstance(var,ModelMatrixSymbol) or isinstance(var,ModelVectorSymbol):
                    if var._matrix not in matries:
                        param_string += f'{var._matrix} = p.{var._matrix};\n\t'
                        matries.append(var._matrix)
                elif isinstance(var,ModelSymbol):
                    param_string += f'{var.name} = p.{var.name};\n\t'
                elif isinstance(var,ModelMatrix) or isinstance(var,ModelVector):
                    param_string += f'{var._matrix_symbol} = p.{var._matrix_symbol};\n\t'
            elif var not in U:
                unknown_vars.append(var)
                if isinstance(var,VarVector):
                    isVarVector = True
                    varVector = var
                print(f'Unknown variable {var} found in function {func_name}. It will be added to the function signature.')

        # split expr into groups
        replacments, exprs = sym.cse(expr,symbols=SymbolInterator())
        if isinstance(expr,tuple):
            expr = tuple(exprs)
        elif isinstance(expr,list):
            expr = exprs
        else:
            expr = exprs[0]      

        group_string = '%% create common groups\n\t'
        for variable, expression in replacments:
            group_string +=f'{variable} = {sym.printing.octave.octave_code(expression, user_functions=user_functions)};\n\t'
        
        # convert to octave string and covert states to vector form
        if isVarVector:
            if isinstance(expr,sym.matrices.dense.MutableDenseMatrix):
                for i in range(len(expr)):
                    if isinstance(expr[i],sym.core.numbers.Number) or isinstance(expr[i],sym.core.numbers.One):
                        expr[i] = varVector[0]*OctaveZero('z') + expr[i]                
        out = '%% create output vector\n\tout = ' + sym.printing.octave.octave_code(expr, user_functions=user_functions)

        #convert state vector calls
        my_replace = lambda x: f'U({int(x.group(1))+1})'
        out = re.sub(r"u_(?P<index>\d+)",my_replace,out)
        group_string = re.sub(r"u_(?P<index>\d+)",my_replace,group_string)

        # make the file pretty...
        out = out.replace(';',';...\n\t\t')

        file_sig = f'%{func_name.upper()} Auto-generated function from moyra\n\t'
        file_sig += f'%\n\t'
        file_sig += f'%\tCreated at : {ctime(time())} \n\t'
        file_sig += f'%\tCreated with : moyra https://pypi.org/project/moyra/\n\t'
        file_sig += f'%\n\t'


        # wrap output in octave function signature
        # create unknow var string
        unknown_str = '' if not unknown_vars else ','+','.join(sorted([str(i) for i in unknown_vars], key=str))
        signature = f'function out = {func_name}(p,U{unknown_str})\n\t'
        

        ## tidy params with curly braces in name
        my_replace = lambda x: f'_{x.group(1)}'
        param_string = re.sub(r"_\{(?P<index>.+)\}",my_replace,param_string)
        group_string = re.sub(r"_\{(?P<index>.+)\}",my_replace,group_string)
        out = re.sub(r"_\{(?P<index>.+)\}",my_replace,out)

        octave_string = signature + file_sig + param_string + group_string + out + ';\nend'
        return octave_string






