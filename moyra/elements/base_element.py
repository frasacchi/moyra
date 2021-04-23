class BaseElement:
    def calc_ke(self,p):
        # return a sybmolic equation for the Kinetic Energy of the element
        raise NotImplementedError("calc_ke not implemented in the current element")
    def calc_pe(self,p):
        # return a sybmolic equation for the Potential Energy of the element
        raise NotImplementedError("calc_pe not implemented in the current element")
    def calc_rdf(self,p):
        # return a sybmolic equation for the Rayleigh Dissaption Function
        #  of the element
        raise NotImplementedError("calc_rdf (Rayleigh Dissapative Function) not implemented in the current element")
