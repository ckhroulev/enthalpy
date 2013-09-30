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

def generic_equations(J, A, R, pism=False):
    """Assemble matrix rows corresponding to the interior."""
    # generic equation coefficients:
    for k in xrange(1, J-1):
        if pism == True:
            Rminus = 0.5 * (R[k-1] + R[k])
            Rplus = 0.5 * (R[k] + R[k+1])

            A[k, k-1] = -Rminus
            A[k, k]   = 1.0 + (Rminus + Rplus)
            A[k, k+1] = -Rplus
        else:
            A[k, k-1] = -R[k]
            A[k, k]   = 1.0 + 2.0 * R[k]
            A[k, k+1] = -R[k]

def step(J, T, A, R, dt, dz, n_steps, T_surface, G=None, T_basal=None, pism=False):
    """Set boundary conditions and step."""
    for k in xrange(n_steps):
        b = T.copy()
        
        # set the top boundary condition (Dirichlet)
        A[J-1, J-1] = 1.0
        b[J-1] = T_surface

        # set the basal boundary condition
        if G is not None:
            if pism == True:
                Rminus = R[0]
                Rplus  = R[1]
                A[0, 0] = 1.0 + (Rminus + Rplus)
                A[0, 1] = - (Rminus + Rplus)
            else:
                A[0, 0] = 1.0 + 2.0 * R[0]
                A[0, 1] = -2.0 * R[0]

            b[0] += 2 * dt * G / dz
        else:
            A[0,0]  = 1.0
            A[0,1]  = 0.0
            b[0]    = T_basal

        # solve the system
        T = linalg.spsolve(sp.csr_matrix(A), b)

    return A, b, T

def solve(pism=False):
    H = 1000.0                     # meters
    J = 41                        # number of grid points
    dz = H / (J - 1)

    dt        = convert(10, "years", "seconds")
    G         = convert(42, "mW m-2", "W m-2")
    T_surface = convert(-30.0, "Celsius", "K")

    rho = 910.0                # density
    c   = 2009.0               # heat capacity
    k   = 2.10                 # conductivity
    K   = k / (c * rho)
    R_i   = K * dt / dz**2
    ratio = 0.01

    # set the initial temperature distribution
    z = np.linspace(0, H, J)
    T = G / k * (H - z) + T_surface

    R = np.zeros_like(z) + R_i
    R[z < 200] *= ratio

    A = sp.lil_matrix((J,J))

    generic_equations(J, A, R, pism)

    T_basal = T[0] - 10.0

    plt.figure()
    plt.hold(True)
    plt.plot(z, T, color='red')

    for n in range(100):
        # run with Dirichlet basal B.C. for a while:
        A, b, T = step(J, T, A, R, dt, dz, 100, T_surface, T_basal=T_basal)
        plt.plot(z, T, color="blue")

    basal_flux = -(T[1] - T[0]) / dz * k
    print "basal flux=%f" % basal_flux

    # run with non-zero geothermal flux:
    for n in range(0):
        A, b, T = step(J, T, A, R, dt, dz, 50, T_surface, G=basal_flux)
        plt.plot(z, T, color='black')

    plt.grid()
    plt.xlabel("z, meters above base")
    plt.ylabel("T, Kelvin")

    return A, b, T
    
A1, b1, T = solve(pism=False)

A2, b2, T = solve(pism=True)


plt.show()
