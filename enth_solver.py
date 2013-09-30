import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import udunits2 as U
import matplotlib.pyplot as plt

def convert(value, str1, str2):
    s = U.System()

    u1 = U.Unit(s, str1)
    u2 = U.Unit(s, str2)
    c = U.Converter((u1, u2))

    return c(value)
    
def generic_equations(A, R):
    """Assemble matrix rows corresponding to the interior."""

    J = A.shape[0]
    # generic equation coefficients:
    for k in xrange(1, J-1):
        Rminus = 0.5 * (R[k-1] + R[k])
        Rplus = 0.5 * (R[k] + R[k+1])

        A[k, k-1] = -Rminus
        A[k, k]   = 1.0 + (Rminus + Rplus)
        A[k, k+1] = -Rplus

    return A

def step(E, A, R, dt, dz, n_steps, E_surface, G=None, E_basal=None):
    """Set boundary conditions and step."""

    J = A.shape[0]

    for k in xrange(n_steps):
        b = E.copy()
        
        # set the top boundary condition (Dirichlet)
        A[J-1, J-1] = 1.0
        b[J-1] = E_surface

        # set the basal boundary condition
        if G is not None:
            Rminus = R[0]
            Rplus  = R[1]
            A[0, 0] = 1.0 + (Rminus + Rplus)
            A[0, 1] = - (Rminus + Rplus)

            K0 = 1
            b[0] += 2 * (dt * K0 / dz**2) * dz * G / K0
        else:
            A[0,0]  = 1.0
            A[0,1]  = 0.0
            b[0]    = E_basal

        # solve the system
        E = linalg.spsolve(sp.csr_matrix(A), b)

    return A, b, E

def solve(J=21):
    global K
    H = 1000.0                     # meters
    dz = H / (J - 1)

    dt        = convert(10, "years", "seconds")
    G         = convert(42, "mW m-2", "W m-2")
    E_surface = 40180.0         # J/kg

    rho = 910.0                # ice density
    c   = 2009.0               # ice specific heat capacity
    k   = 2.10                 # ice thermal conductivity
    K   = k / c
    R_i   = K * dt / dz**2
    ratio = 0.1

    # set the initial temperature distribution
    z = np.linspace(0, H, J)
    E = np.zeros_like(z) + E_surface

    R = np.zeros_like(z) + R_i
    R[z < 200] *= ratio

    A = sp.lil_matrix((J,J))

    A = generic_equations(A, R)

    E_basal = 80360.0

    plt.hold(True)
    plt.plot(z, E, color='red')

    basal_flux = 0.042
    print "basal flux=%f W m-2" % basal_flux

    # run with non-zero geothermal flux:
    for n in range(10):
        A, b, E = step(E, A, R, dt, dz, 500, E_surface, G=basal_flux)
        plt.plot(z, E, color='black')

    plt.grid()
    plt.xlabel("z, meters above base")
    plt.ylabel("E, J/kg")

    print "basal flux=%f W m-2" % (-(E[1] - E[0]) / dz * (K * ratio))

    return A, b, E
    
A, b, E = solve(201)

plt.show()
