from functools import cache
import sympy as sym
from sympy.abc import t
from .base_element import BaseElement


class NonLinearElement(BaseElement):
    """Geometrically nonlinear beam element (PANG formulation).

    Displacement field expanded to 3rd order in modal DOFs:
      w = Wᵀq_w + d·[θ·(1 - ½φ_w²) - ⅙θ³]
      v = Vᵀq_v + d·[-½θ² - ½φ_v² - φ_v·θ·φ_w]
      u = -½(q_wᵀFw q_w + q_vᵀFv q_v)
          + d·[-φ_v·(1 - ½θ² + ½φ_w²) - φ_w·θ]

    where d = b·(1+a), θ = Tᵀq_t, φ_w = W1ᵀq_w, φ_v = V1ᵀq_v,
    Fw(y) = ∫₀ʸ W1·W1ᵀ ds, Fv(y) = ∫₀ʸ V1·V1ᵀ ds.

    Shape functions use scaled/shifted Chebyshev polynomials on [0,L]:
      s(y) = -2y/L + 1   (maps [0,L] → [1,-1])
      T_0=1, T_1=s, T_n = 2s·T_{n-1} - T_{n-2}
      W_i(y) = T_{i-1}(s)·y²   satisfies W(0)=0, W'(0)=0
      V_i(y) = T_{i-1}(s)·y²   satisfies V(0)=0, V'(0)=0
      T_i(y) = T_{i-1}(s)·y    satisfies T(0)=0

    DOF ordering: q = [q_w₁…q_wNw, q_v₁…q_vNv, q_t₁…q_tNt]

    Parameters
    ----------
    q : sympy.Matrix (Nw+Nv+Nt × 1)
        Time-dependent generalised coordinates.
    Nw, Nv, Nt : int
        Number of flapwise, chordwise, torsional modes.
    y : sympy.Symbol
        Spanwise coordinate.
    L : scalar or sympy.Expr
        Beam length.
    density : scalar or sympy.Expr
        Mass per unit length m(y).
    EI1 : scalar or sympy.Expr
        Flapwise bending stiffness.
    EI2 : scalar or sympy.Expr
        Chordwise bending stiffness.
    EI12 : scalar or sympy.Expr
        Bending-bending coupling stiffness (off-diagonal of [EI] matrix).
    GJ : scalar or sympy.Expr
        Torsional stiffness.
    b : scalar or sympy.Expr, optional
        Half-chord b(y). Default 0 (disables nonlinear coupling terms).
    a : scalar or sympy.Expr, optional
        Normalised elastic-axis offset from midchord a(y) in [-1,1].
        d = b*(1+a) is the chordwise offset used in the displacement field.
    mxx : scalar or sympy.Expr, optional
        Torsional (spanwise-axis) mass moment of inertia per unit length.
    myy : scalar or sympy.Expr, optional
        Chordwise-axis mass moment of inertia per unit length.
    mzz : scalar or sympy.Expr, optional
        Flapwise-axis mass moment of inertia per unit length.
    simplify : bool, optional
        Apply trigsimp/expand to the mass matrix. Default True.
    name : str, optional
        Element name for bookkeeping.
    """

    def __init__(self, q, Nw, Nv, Nt, y, L,
                 density, EI1, EI2, EI12, GJ,
                 b=0, a=sym.Integer(0),
                 K=0,
                 mxx=0, myy=0, mzz=0,
                 grav_vec=None,
                 simplify=True, name="default"):
        if Nw + Nv + Nt != len(q):
            raise ValueError(
                f"len(q)={len(q)} must equal Nw+Nv+Nt={Nw+Nv+Nt}"
            )
        self._Nw = Nw
        self._Nv = Nv
        self._Nt = Nt
        self._L = L
        self._y = y
        self._density = density
        self._EI1 = EI1
        self._EI2 = EI2
        self._EI12 = EI12
        self._GJ = GJ
        self._b = b
        self._a = a
        self._K = K
        self._grav_vec = sym.Matrix(grav_vec) if grav_vec is not None else sym.zeros(3, 1)
        self._mxx = mxx
        self._myy = myy
        self._mzz = mzz
        self._simplify = simplify
        super().__init__(q, name)

    Nw = property(lambda self: self._Nw)
    Nv = property(lambda self: self._Nv)
    Nt = property(lambda self: self._Nt)
    L = property(lambda self: self._L)
    y_sym = property(lambda self: self._y)
    density = property(lambda self: self._density)
    EI1 = property(lambda self: self._EI1)
    EI2 = property(lambda self: self._EI2)
    EI12 = property(lambda self: self._EI12)
    GJ = property(lambda self: self._GJ)
    b = property(lambda self: self._b)
    a = property(lambda self: self._a)
    K = property(lambda self: self._K)
    grav_vec = property(lambda self: self._grav_vec)
    mxx = property(lambda self: self._mxx)
    myy = property(lambda self: self._myy)
    mzz = property(lambda self: self._mzz)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                      #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_dot(basis, dofs):
        """(basis.T @ dofs)[0], or 0 when either dimension is zero."""
        if basis.shape[0] == 0 or dofs.shape[0] == 0:
            return sym.Integer(0)
        return (basis.T * dofs)[0]

    @staticmethod
    def _safe_quad(dofs, mat):
        """(dofs.T @ mat @ dofs)[0], or 0 when dofs is empty."""
        if dofs.shape[0] == 0:
            return sym.Integer(0)
        return (dofs.T * mat * dofs)[0]

    # ------------------------------------------------------------------ #
    # Shape functions                                                       #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _cheb_shifted(n, y, L):
        """n Chebyshev polynomials T_0..T_{n-1} with s = -2y/L + 1."""
        if n == 0:
            return []
        s = -2 * (y / L) + 1
        polys = [sym.Integer(1), s]
        for _ in range(2, n):
            polys.append(sym.expand(2 * s * polys[-1] - polys[-2]))
        return polys[:n]

    @property
    @cache
    def W_basis(self):
        """Flapwise shape functions W_i(y) = T_{i-1}(s)·y², i=0..Nw-1."""
        y, L = self._y, self._L
        return sym.Matrix([y**2 * p for p in self._cheb_shifted(self._Nw, y, L)])

    @property
    @cache
    def V_basis(self):
        """Chordwise shape functions V_i(y) = T_{i-1}(s)·y², i=0..Nv-1."""
        y, L = self._y, self._L
        return sym.Matrix([y**2 * p for p in self._cheb_shifted(self._Nv, y, L)])

    @property
    @cache
    def T_basis(self):
        """Torsional shape functions T_i(y) = T_{i-1}(s)·y, i=0..Nt-1."""
        y, L = self._y, self._L
        return sym.Matrix([y * p for p in self._cheb_shifted(self._Nt, y, L)])

    @property
    @cache
    def W1_basis(self):
        return sym.Matrix([f.diff(self._y) for f in self.W_basis])

    @property
    @cache
    def V1_basis(self):
        return sym.Matrix([f.diff(self._y) for f in self.V_basis])

    @property
    @cache
    def W2_basis(self):
        return sym.Matrix([f.diff(self._y, 2) for f in self.W_basis])

    @property
    @cache
    def V2_basis(self):
        return sym.Matrix([f.diff(self._y, 2) for f in self.V_basis])

    @property
    @cache
    def T1_basis(self):
        return sym.Matrix([f.diff(self._y) for f in self.T_basis])

    # ------------------------------------------------------------------ #
    # Axial shortening integrals                                           #
    # ------------------------------------------------------------------ #

    @property
    @cache
    def Fw(self):
        """∫₀ʸ W1(s)·W1(s)ᵀ ds  (Nw×Nw, polynomial in y)."""
        y_sym = self._y
        s = sym.Dummy('s_int')
        W1_s = sym.Matrix([f.subs(y_sym, s) for f in self.W1_basis])
        return sym.integrate(W1_s * W1_s.T, (s, 0, y_sym))

    @property
    @cache
    def Fv(self):
        """∫₀ʸ V1(s)·V1(s)ᵀ ds  (Nv×Nv, polynomial in y)."""
        y_sym = self._y
        s = sym.Dummy('s_int')
        V1_s = sym.Matrix([f.subs(y_sym, s) for f in self.V1_basis])
        return sym.integrate(V1_s * V1_s.T, (s, 0, y_sym))

    # ------------------------------------------------------------------ #
    # Displacement field                                                    #
    # ------------------------------------------------------------------ #

    @property
    @cache
    def displacement(self):
        """PANG nonlinear displacement vector [u, v, w] as sympy.Matrix(3,1).

        Represents the displacement of the reference point located at
        chordwise offset d = b*(1+a) from the elastic axis.
        The axial component u includes geometric shortening.
        """
        Nw, Nv = self._Nw, self._Nv
        q_w = self.q[:Nw, :]
        q_v = self.q[Nw:Nw + Nv, :]
        q_t = self.q[Nw + Nv:, :]

        w_m   = self._safe_dot(self.W_basis,  q_w)   # elastic-axis flapwise disp.
        v_m   = self._safe_dot(self.V_basis,  q_v)   # elastic-axis chordwise disp.
        theta = self._safe_dot(self.T_basis,  q_t)   # twist angle
        phi_w = self._safe_dot(self.W1_basis, q_w)   # flapwise slope dw/dy
        phi_v = self._safe_dot(self.V1_basis, q_v)   # chordwise slope dv/dy

        d = self._b * (1 + self._a)

        # Flapwise displacement (3rd-order expansion of cross-section rotation)
        w = (w_m
             + d * (theta * (1 - sym.Rational(1, 2) * phi_w**2)
                    - sym.Rational(1, 6) * theta**3))

        # Chordwise displacement (change from reference chordwise position)
        v = (v_m
             + d * (- sym.Rational(1, 2) * theta**2
                    - sym.Rational(1, 2) * phi_v**2
                    - phi_v * theta * phi_w))

        # Axial displacement: geometric shortening + chordwise coupling
        u_short = -sym.Rational(1, 2) * (
            self._safe_quad(q_w, self.Fw) + self._safe_quad(q_v, self.Fv)
        )
        u = (u_short
             + d * (- phi_v * (1 - sym.Rational(1, 2) * theta**2
                               + sym.Rational(1, 2) * phi_w**2)
                    - phi_w * theta))

        return sym.Matrix([u, v, w])

    # ------------------------------------------------------------------ #
    # Mass matrix                                                           #
    # ------------------------------------------------------------------ #

    @property
    @cache
    def M(self):
        """Nonlinear mass matrix M(q) from Jacobian of displacement field.

        Assembles translational inertia via:
          M_trans = ∫₀ᴸ ρ(y) · (∂r/∂q)ᵀ · (∂r/∂q) dy

        Adds diagonal rotary-inertia blocks (linearised angular velocities):
          M_mxx from torsional rate  θ̇ = T1ᵀ q̇_t
          M_mzz from flapwise slope  φ̇_w = W1ᵀ q̇_w
          M_myy from chordwise slope φ̇_v = V1ᵀ q̇_v
        """
        y, L = self._y, self._L
        Nw, Nv, Nt = self._Nw, self._Nv, self._Nt
        Ntot = Nw + Nv + Nt

        # Translational contribution
        J = self.displacement.jacobian(self.q)   # 3 × Ntot
        M = sym.integrate(self._density * J.T * J, (y, 0, L))

        # Embed a short vector into the full DOF space at given offset
        def _embed(vec, offset):
            out = sym.zeros(Ntot, 1)
            for i, vi in enumerate(vec):
                out[offset + i] = vi
            return out

        # Torsional rotary inertia (about spanwise axis)
        if isinstance(self._mxx, sym.Expr) or self._mxx != 0:
            T1f = _embed(self.T1_basis, Nw + Nv)
            M += sym.integrate(self._mxx * T1f * T1f.T, (y, 0, L))

        # Flapwise-slope rotary inertia (about chordwise axis)
        if isinstance(self._mzz, sym.Expr) or self._mzz != 0:
            W1f = _embed(self.W1_basis, 0)
            M += sym.integrate(self._mzz * W1f * W1f.T, (y, 0, L))

        # Chordwise-slope rotary inertia (about flapwise axis)
        if isinstance(self._myy, sym.Expr) or self._myy != 0:
            V1f = _embed(self.V1_basis, Nw)
            M += sym.integrate(self._myy * V1f * V1f.T, (y, 0, L))

        return M

    # ------------------------------------------------------------------ #
    # Potential energy                                                      #
    # ------------------------------------------------------------------ #

    @property
    @cache
    def elastic_pe(self):
        """Structural strain energy (linear Euler-Bernoulli + torsion):
          U = ½ ∫₀ᴸ [EI1·κ_w² + 2·EI12·κ_w·κ_v + EI2·κ_v²
                      + GJ·τ² - 2·K·κ_w·τ] dy
        where κ_w = d²w/dy², κ_v = d²v/dy², τ = dθ/dy.
        K is the bending-torsion stiffness coupling (Bisplinghoff sign: U_K = -K·κ·τ).
        """
        y, L = self._y, self._L
        Nw, Nv = self._Nw, self._Nv
        q_w = self.q[:Nw, :]
        q_v = self.q[Nw:Nw + Nv, :]
        q_t = self.q[Nw + Nv:, :]

        kappa_w = self._safe_dot(self.W2_basis, q_w)
        kappa_v = self._safe_dot(self.V2_basis, q_v)
        tau     = self._safe_dot(self.T1_basis, q_t)

        U = sym.Integer(0)
        if isinstance(self._EI1, sym.Expr) or self._EI1 != 0:
            U += sym.Rational(1, 2) * self._EI1 * kappa_w**2
        if isinstance(self._EI2, sym.Expr) or self._EI2 != 0:
            U += sym.Rational(1, 2) * self._EI2 * kappa_v**2
        if isinstance(self._EI12, sym.Expr) or self._EI12 != 0:
            U += self._EI12 * kappa_w * kappa_v
        if isinstance(self._GJ, sym.Expr) or self._GJ != 0:
            U += sym.Rational(1, 2) * self._GJ * tau**2
        if isinstance(self._K, sym.Expr) or self._K != 0:
            U -= self._K * kappa_w * tau

        return sym.integrate(U, (y, 0, L))

    @property
    @cache
    def grav_pe(self):
        """Gravitational PE from beam-relative displacement:
          PE_grav = -∫₀ᴸ ρ(y) · (g_vec · r(y,q)) dy

        r(y,q) is the PANG nonlinear displacement field [u,v,w]. The Euler-
        Lagrange derivative -∂PE_grav/∂q automatically produces the PANG
        gravity generalised-force vector, including all nonlinear corrections
        (bending-torsion-gravity coupling via the b*(1+a) offset terms).

        Note: this is the beam-deformation contribution only. The rigid-body
        gravity term (-m_total·g·q_plunge) must be handled separately via the
        body-frame elements (RigidElement wing sections etc.).
        """
        if (self._grav_vec.T * self._grav_vec)[0] == 0:
            return sym.Integer(0)
        integrand = -(self._grav_vec.T * self.displacement)[0] * self._density
        return sym.integrate(integrand, (self._y, 0, self._L))

    @property
    @cache
    def pe(self):
        return self.elastic_pe + self.grav_pe

    @property
    def rdf(self):
        return sym.Integer(0)

    @property
    @cache
    def ke(self):
        if self._density == 0:
            return sym.Integer(0)
        T = sym.Rational(1, 2) * self.qd.T * self.M * self.qd
        return T[0].expand()

    # ------------------------------------------------------------------ #
    # Static helper: access shape functions without creating an element    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def ShapeFunctions(Nw, Nv, Nt, y, L):
        """PANG Chebyshev shape function vectors and derivatives.

        Parameters
        ----------
        Nw, Nv, Nt : int
            Number of flapwise, chordwise, torsional modes.
        y : sympy.Symbol
            Spanwise coordinate.
        L : scalar or sympy.Expr
            Beam length.

        Returns
        -------
        W, V, T   : Matrix (N×1)  shape function vectors
        W1, V1    : Matrix (N×1)  first derivatives w.r.t. y
        W2, V2    : Matrix (N×1)  second derivatives w.r.t. y
        T1        : Matrix (N×1)  first derivative of torsion functions
        """
        def _cheb(n):
            if n == 0:
                return []
            s = -2 * (y / L) + 1
            polys = [sym.Integer(1), s]
            for _ in range(2, n):
                polys.append(sym.expand(2 * s * polys[-1] - polys[-2]))
            return polys[:n]

        W  = sym.Matrix([y**2 * p for p in _cheb(Nw)])
        V  = sym.Matrix([y**2 * p for p in _cheb(Nv)])
        T  = sym.Matrix([y     * p for p in _cheb(Nt)])
        W1 = sym.Matrix([f.diff(y)    for f in W])
        V1 = sym.Matrix([f.diff(y)    for f in V])
        W2 = sym.Matrix([f.diff(y, 2) for f in W])
        V2 = sym.Matrix([f.diff(y, 2) for f in V])
        T1 = sym.Matrix([f.diff(y)    for f in T])
        return W, V, T, W1, V1, W2, V2, T1
