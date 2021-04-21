from . import ExternalForce

class CustomForce(ExternalForce):

    def __init__(self,Forcingfunction):
        self.q_func = Forcingfunction

    def Q(self):
        return None
    
    def subs(self,*args):
        return self

    def msubs(self,*args):
        return self

    def cancel(self):
        return self

    def expand(self):
        return self

    def integrate(self,*args):
        return self

    def linearise(self,x,x_f):
        return self

    def lambdify(self,params):
        return self.q_func   

        
