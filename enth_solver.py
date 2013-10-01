import scipy.sparse as sp
import numpy as np
from scipy.sparse import linalg
import udunits2 as U
import matplotlib.pyplot as plt

L   = 3.34e5                    # latent heat of fusion
rho = 910.0                     # ice density
c   = 2009.0                    # ice specific heat capacity
k   = 2.10                      # ice thermal conductivity
K   = k / c                     # enthalpy diffusivity

W = 0.01                        # maximum allowed water fraction

T_0         = 223.15            # reference temperature
g           = 9.81              # standard gravity
T_melting_0 = 273.15            # ice melting temperature at p=0
beta        = 7.9e-8            # Clausius-Clapeyron constant
ratio       = 0.1               # enthalpy conductivity ratio

K0 = K * ratio

def convert(value, str1, str2):
    "Unit converter"
    s = U.System()

    u1 = U.Unit(s, str1)
    u2 = U.Unit(s, str2)
    c = U.Converter((u1, u2))

    return c(value)

def T_melting(p):
    "Pressure-melting temperature as a function of pressure."
    global beta, T_melting_0
    return T_melting_0 - beta * p

def P(depth):
    "Pressure as a function of depth."
    global rho, g
    return rho * g * depth
    
def Ects(p):
    "Enthalpy corresponding to CTS with zero water fraction."
    global c, T_0
    return c * (T_melting(p) - T_0)

def compute_R(E, z, R_ice, ratio):
    "Compute the R coefficient"

    S = z[-1]
    R = np.zeros_like(E) + R_ice
    E_cts = Ects(P(S - z))

    R[E > E_cts] *= ratio

    return R

def Dc(G, E_surface):
    "Theoretical depth of the cold layer."
    global c, K, T_0, T_melting_0, beta, g, rho
    return -(c*K*T_0 + (E_surface - c*T_melting_0)*K)/(beta*c*g*rho*K + G)

def E_exact(G, E_surface, K, K0, S, z):
    "Exact solution for the problem."
    D_cold = Dc(G, E_surface)
    D_temp = S - D_cold

    E = np.zeros_like(z)

    depth = S - z

    E_transition = E_surface + D_cold * G / K

    E[z >  D_temp] = E_surface + (G / K) * depth[z > D_temp]
    E[z <= D_temp] = E_transition + (G / K0) * (depth[z <= D_temp] - D_cold)
    
    return E
    
def step(E, A, z, dt, dz, E_surface, R_ice=1, G=None, E_basal=None, clip=False):
    """Set boundary conditions and step."""
    global ratio

    J = A.shape[0]
    b = E.copy()

    R = compute_R(E, z, R_ice, ratio)
    
    # basal boundary condition:
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

    # interior of the column:
    for k in xrange(1, J-1):
        Rminus = 0.5 * (R[k-1] + R[k])
        Rplus = 0.5 * (R[k] + R[k+1])

        A[k, k-1] = -Rminus
        A[k, k]   = 1.0 + (Rminus + Rplus)
        A[k, k+1] = -Rplus

    # top boundary condition (Dirichlet)
    A[J-1, J-1] = 1.0
    b[J-1] = E_surface

    # solve the system
    E = linalg.spsolve(sp.csr_matrix(A), b)

    if clip == True:
        depth = z[-1] - z
        E_CTS_max = Ects(P(depth)) + W * L
        
        E[E > E_CTS_max] = E_CTS_max[E > E_CTS_max]
    
    return A, b, E, R

def solve(J=21, Dc=200, S=1000.0, T=10e3, G=0.042, E_surface=40180.0, dt=10.0,
          clip=False, verbose=False):

    dt = convert(dt, "years", "seconds")
    T = convert(T, "years", "seconds")

    n_steps = int(float(T) / dt)

    if verbose:
        print "will take %d steps %f years long (%f years total)" % (n_steps,
                                                                     convert(dt, "seconds", "years"),
                                                                     convert(T, "seconds", "years"))
    dz = S / (J - 1)

    if verbose:
        print "using %d grid levels (dz=%f meters)" % (J, dz)

    R_ice   = K * dt / dz**2
    ratio = 0.1

    # set the initial enthalpy distribution
    z = np.linspace(0, S, J)
    E = np.zeros_like(z) + E_surface

    A = sp.lil_matrix((J,J))

    plt.hold(True)

    if verbose:
        print "given basal flux=%f W m-2" % G

    # time-stepping loop
    for n in range(n_steps):
        A, b, E, R = step(E, A, z, dt, dz, E_surface, R_ice=R_ice, G=G, clip=clip)

    if verbose:
        K0 = R[0] * dz**2 / dt
        print "computed basal flux=%f W m-2" % (-(E[1] - E[0]) / dz * K0)

        Ks = R[-1] * dz**2 / dt
        print "computed surface flux=%f W m-2" % (-(E[-1] - E[-2]) / dz * Ks)

    return A, b, E, z, R

def test(J=41, verbose=False, plot=False):
    G = 0.0003
    E_surface = 1e5
    S = 1000.0
    A, b, E, z, R = solve(J, S=S, G=G, dt=100, T=1e4, E_surface=E_surface, verbose=verbose)


    depth = S - z
    E_CTS = Ects(P(depth))

    try:
        Dt = z[E > E_CTS].max()
    except:
        Dt = 0

    Dt_theory = S - Dc(G, E_surface)
    exact = E_exact(G, E_surface, K, K0, S, z)

    error = np.abs(E - exact).max()

    if verbose:
        print "Depth of the temperate layer: %f meters" % Dt
        print "Depth of the temperate layer (theory): %f meters" % Dt_theory


        print "temperate layer depth error: %f meters" % np.abs(Dt - Dt_theory)
        print "max. enthalpy error: %f J/kg" % error

    if plot == True:
        plt.plot(z, E, color='black')
        plt.grid()
        plt.xlabel("z, meters above base")
        plt.ylabel("E, J/kg")

        plt.plot(z, E_CTS, label="CTS enthalpy")
        plt.plot(z, E_CTS + W * L, label="max. allowed enthalpy")

        ybounds = plt.axes().get_ybound()

        plt.plot([Dt_theory, Dt_theory], ybounds, '--', color="black",
                 label="exact CTS")


        plt.plot(z, exact, '--', color="red", label="exact solution")

        plt.legend()
        plt.show()

    return error

J = 11
Js = []
errors = []
for F in xrange(1, 12):
    J = J * 2
    print "J = %d" % J
    Js.append(J)
    errors.append(test(J))

plt.hold(True)
plt.loglog(Js, 1000.0/np.array(Js))
plt.loglog(Js, errors)
plt.xlabel("J")
plt.ylabel("max(error)")
plt.show()
