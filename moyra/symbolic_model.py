import re
import pickle
import sympy as sym
import numpy as np
from scipy.linalg import eig
import sympy.physics.mechanics as me
from sympy.physics.vector.printing import vpprint, vlatex
from sympy.abc import x,y,t
from .helper_funcs import linearise_matrix
from .numeric_model import NumericModel
import pickle
from .forces import ZeroForce
from .printing import model as print_model
from .model_parameters import ModelSymbol, ModelMatrix,ModelMatrixSymbol, ModelValue
from time import time, ctime

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
    def FromElementsAndForces(cls,FwtParams,Elements,ExtForces = None):
        """
        Create a symbolic Model instance from a set Elements and external forces
        """
        p = FwtParams 

        # Calc K.E, P.E and Rayleigh Dissaptive Function
        T = U = D = sym.Integer(0)
        # add K.E for each Rigid Element
        for ele in Elements:
            T += ele.calc_ke(p)
            U += ele.calc_pe(p)
            D += ele.calc_rdf(p)

        # calculate EoM
        Lag = sym.Matrix([T-U])
        D = sym.Matrix([D])
        term_1 = Lag.jacobian(p.qd).diff(me.dynamicsymbols._t).T.expand()
        term_2 = Lag.jacobian(p.q).T
        term_3 = D.jacobian(p.qd).T

        # Get Mass Matrix and 'internal' forcing term
        M = term_1.jacobian(p.qdd) # assuming all parts in term 1 contribute only to mass matrix
        f = sym.expand(term_1-M*p.qdd) - term_2 + term_3
        return cls(M,f,T,U,ExtForces)


    def __init__(self,M,f,T,U,ExtForces = None):
        """Initialise a Symbolic model of the form 
        $M\ddot{q}+f(\dot{q},q,t)-ExtForces(\dot{q},q,t) = 0$

        with the Symbolic Matricies M,f,and Extforces
        """
        self.M = M
        self.f = f
        self.T = T
        self.U = U
        if ExtForces is None:
            self.ExtForces = ZeroForce(f.shape[0])
        else:
            self.ExtForces = ExtForces

    def cancel(self):
        """
        Creates a new instance of a Symbolic model with the cancel simplifcation applied
        """
        ExtForces = self.ExtForces.cancel() if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else sym.cancel(self.T)
        U = self.U if isinstance(self.U,int) else sym.cancel(self.U)
        return SymbolicModel(sym.cancel(self.M),sym.cancel(self.f),
                            T,U,ExtForces)

    def expand(self):
        """
        Creates a new instance of a Symbolic model with the cancel simplifcation applied
        """
        ExtForces = self.ExtForces.expand() if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else sym.expand(self.T)
        U = self.U if isinstance(self.U,int) else sym.expand(self.U)
        return SymbolicModel(sym.expand(self.M),sym.expand(self.f),
                            T,U,ExtForces)


    def subs(self,*args):
        """
        Creates a new instance of a Symbolic model with the substutions supplied
         in args applied to all the Matricies
        """
        ExtForces = self.ExtForces.subs(*args) if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else self.T.subs(*args)
        U = self.U if isinstance(self.U,int) else self.U.subs(*args)
        return SymbolicModel(self.M.subs(*args),self.f.subs(*args),
                            T,U,ExtForces)

    def msubs(self,*args):
        """
        Creates a new instance of a Symbolic model with the substutions supplied
         in args applied to all the Matricies
        """
        ExtForces = self.ExtForces.msubs(*args) if self.ExtForces is not None else None

        # handle zero kinetic + pot energies
        T = self.T if isinstance(self.T,int) else me.msubs(self.T,*args)
        U = self.U if isinstance(self.U,int) else me.msubs(self.U,*args)
        return SymbolicModel(me.msubs(self.M,*args),me.msubs(self.f,*args),
                            T,U,ExtForces)

    def linearise(self,p):
        """
        Creates a new instance of the symbolic model class in which the EoM have been 
        linearised about the fixed point p.q_0
        """
        # Calculate Matrices at the fixed point
        # (go in reverse order so velocitys are subbed in before positon)

        # get the full EoM's for free vibration and linearise
        eom = self.M*p.qdd + self.f

        x = [ j for i in range(len(p.q)) for j in [p.q[i],p.qd[i],p.qdd[i]]]
        fp = [ j for i in range(len(p.q)) for j in [p.fp[::2][i],p.fp[1::2][i],0]]
        eom_lin = linearise_matrix(eom,x,fp)

        #extract linearised M
        M_lin = eom_lin.jacobian(p.qdd)

        #extract linerised f
        f_lin = (eom_lin - M_lin*p.qdd).doit().expand()

        # Linearise the External Forces
        extForce_lin = self.ExtForces.linearise(p.x,p.fp) if self.ExtForces is not None else None

        # create the linearised model and return it
        return SymbolicModel(M_lin,f_lin,0,0,extForce_lin)

    def extract_matrices(self,p):
        """
        From the current symbolic model extacts the classic matrices A,B,C,D,E as per the equation below
        A \ddot{q} + B\dot{q} + Cq = D\dot{q} + Eq
        """
        A = self.M
        B = self.f.jacobian(p.qd)
        C = self.f.jacobian(p.q)
        D = self.ExtForces.Q().jacobian(p.qd)
        E = self.ExtForces.Q().jacobian(p.q)
        return A,B,C,D,E

    def free_body_eigen_problem(self,p):
        """
        gets the genralised eigan matrices for the free body problem.
        They are of the form:
            |   I   0   |       |    0    I   |
        M=  |   0   M   |   ,K= |   -C   -B   |
        such that scipy.linalg.eig(K,M) solves the problem 

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        M = sym.eye(p.qs*2)
        M[-p.qs:,-p.qs:]=self.M

        K = sym.zeros(p.qs*2)
        K[:p.qs,-p.qs:] = sym.eye(p.qs)
        K[-p.qs:,:p.qs] = -self.f.jacobian(p.q)
        K[-p.qs:,-p.qs:] = -self.f.jacobian(p.qd)
        return K,M

    def gen_eigen_problem(self,p):
        """
        gets the genralised eigan matrices for use in solving the frequencies / modes. 
        They are of the form:
            |   I   0   |       |    0    I   |
        M=  |   0   M   |   ,K= |   E-C  D-B  |
        such that scipy.linalg.eig(K,M) solves the problem 

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        M_prime = sym.eye(p.qs*2)
        M_prime[-p.qs:,-p.qs:]=self.M

        _Q = self.ExtForces.Q() if self.ExtForces is not None else sym.Matrix([0]*p.qs)

        #f = (_Q-self.f)

        K_prime = sym.zeros(p.qs*2)
        K_prime[:p.qs,-p.qs:] = sym.eye(p.qs)
        K_prime[-p.qs:,:p.qs] = _Q.jacobian(p.q)-self.f.jacobian(p.q)
        K_prime[-p.qs:,-p.qs:] = _Q.jacobian(p.qd)-self.f.jacobian(p.qd)

        return K_prime, M_prime

    def gen_lin_eigen_problem(self,p):
        """
        gets the genralised eigan matrices for use in solving the frequencies / modes. 
        They are of the form:
            |   I   0   |       |    0    I   |
        M=  |   0   M   |   ,K= |   E-C  D-B  |
        such that scipy.linalg.eig(K,M) solves the problem 

        THE SYSTEM MUST BE LINEARISED FOR THIS TO WORK
        """
        x = [ j for i in range(len(p.q)) for j in [p.q[i],p.qd[i],p.qdd[i]]]
        fp = [ j for i in range(len(p.q)) for j in [p.fp[::2][i],p.fp[1::2][i],0]]

        x_subs = {x[i]:fp[i] for i in range(len(x))}        

        _Q = self.ExtForces.Q() if self.ExtForces is not None else sym.Matrix([0]*p.qs)

        f = (_Q - self.f)

        A = me.msubs(self.M,x_subs)
        B = me.msubs(self._jacobian(f,p.qd),x_subs)
        C = me.msubs(self._jacobian(f,p.q),x_subs)

        M_prime = sym.eye(p.qs*2)
        M_prime[-p.qs:,-p.qs:]=A

        K_prime = sym.zeros(p.qs*2)
        K_prime[:p.qs,-p.qs:] = sym.eye(p.qs)
        K_prime[-p.qs:,:p.qs] = C
        K_prime[-p.qs:,-p.qs:] = B

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
        #Combine and add import statements
        full_code = "from sympy import *\nimport moyra as ma\n\n"+M_code+f_code+T_code+U_code+Q_code+p_code

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
        ExtForce = ExternalForce(_Q)
        return (cls(M,f,T,U,ExtForce),p)
    
    def to_matlab_file(self,p,file_dir):
        funcs = (('get_M',self.M),('get_f',self.f),('get_Q',self.ExtForces.Q()))
        for func_name,matrix in funcs:
            with open(file_dir+f"{func_name}.m",'w') as file:
                file.write(self._gen_octave(matrix,p,func_name))
        p.to_matlab_class(file_dir=file_dir)

    def to_matlab_file_linear(self,p,file_dir):
        mats = self.extract_matrices(p)
        names = ['A','B','C','D','E']
        funcs = list(zip([f'get_{i}' for i in names],mats))
        for func_name,matrix in funcs:
            with open(file_dir+f"{func_name}.m",'w') as file:
                file.write(self._gen_octave(matrix,p,func_name))
        p.to_matlab_class(file_dir=file_dir)
        
    def _gen_octave(self,expr,p,func_name):
        # convert states to non-time dependent variable
        U = sym.Matrix(sym.symbols(f'u_:{p.qs*2}'))
        l = dict(zip(p.x,U))
        expr = me.msubs(expr,l)

        # get parameter replacements
        param_string = '%% extract required parameters from structure\n\t'
        matries = []
        for var in expr.free_symbols:
            if isinstance(var,ModelValue):
                if isinstance(var,ModelMatrixSymbol):
                    if var._matrix not in matries:
                        param_string += f'{var._matrix} = p.{var._matrix};\n\t'
                        matries.append(var._matrix)
                elif isinstance(var,ModelSymbol):
                    param_string += f'{var.name} = p.{var.name};\n\t'
                elif isinstance(var,ModelMatrix):
                    param_string += f'{var._matrix_symbol} = p.{var._matrix_symbol};\n\t'


        # split expr into groups
        replacments, exprs = sym.cse(expr,symbols=(sym.Symbol(f'rep_{i}')for i in range(10000)))
        if isinstance(expr,tuple):
            expr = tuple(exprs)
        elif isinstance(expr,list):
            expr = exprs
        else:
            expr = exprs[0]      

        group_string = '%% create common groups\n\t'
        for variable, expression in replacments:
            group_string +=f'{variable} = {sym.printing.octave.octave_code(expression)};\n\t'
        
        # convert to octave string and covert states to vector form
        out = '%% create output vector\n\tout = ' + sym.printing.octave.octave_code(expr)

        #convert state vector calls
        my_replace = lambda x: f'U({int(x.group(1))+1})'
        out = re.sub(r"u_(?P<index>\d+)",my_replace,out)
        group_string = re.sub(r"u_(?P<index>\d+)",my_replace,group_string)

        # make the file pretty...
        out = out.replace(',',',...\n\t\t').replace(';',';...\n\t\t')

        file_sig = f'%{func_name.upper()} Auto-generated function from moyra\n\t'
        file_sig += f'%\n\t'
        file_sig += f'%\tCreated at : {ctime(time())} \n\t'
        file_sig += f'%\tCreated with : moyra https://pypi.org/project/moyra/\n\t'
        file_sig += f'%\n\t'


        # wrap output in octave function signature
        signature = f'function out = {func_name}(U,p)\n\t'
        octave_string = signature + file_sig + param_string + group_string + out + ';\nend'
        return octave_string






