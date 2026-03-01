from netgen.occ import *
from ngsolve import *
from ngsolve.webgui import Draw
import numpy as np
import time as time
import matplotlib.pyplot as plt
from ngsolve.krylovspace import GMRes

def BulkModulus(G, nu):
    return 2*G*(1 + nu) / (3 - 6*nu)

def YoungsModulus(G, nu):
    return 2*G*(1 + nu)

def LameParameter(G, nu):
    return 2*G*nu / (1 - 2*nu)

def ConfinedCompression(G, nu, viscosity, alpha, n, k, chi, rho, dt, dt_max, dt_growth, R, H, u_max , compression_time, t_end, h, order):
    """
    Simulate 3D confined compression of a poroelastic cylinder using the finite element method.
    
    Parameters
    ----------
    G : float 
        Shear modulus of the material.
    nu : float
        Poisson's ratio of the material.
    viscosity: float
        Viscosity of the fluid in the pores.
    alpha : float
        Biot's coefficient.
    n : float
        Porosity of the material.
    k : float
        Permeability of the material.
    chi : float
        Compressibility of the fluid.
    rho : float
        Density of the fluid.
    dt : float
        Time step size.
    dt_max : float
        Maximum time step size for dynamic time stepping.
    dt_growth : float
        Growth factor for dynamic time stepping.
    R : float
        Radius of the cylindrical sample.
    H : float
        Height of the cylindrical sample.
    u_max : float
        Maximum displacement of the compression.
    compression_time : float
        Time duration for which the compression is applied.
    t_end : float
        Total simulation time.
    h : float
        Maximum mesh element size (Non Dimensional).
    order : int
        Polynomial order for the finite element space.
    
    Returns
    -------
    time_vals : list
        List of time values at which forces were recorded.
    F_solid : list
        List of solid forces recorded at each time step.
    F_fluid : list
        List of fluid forces recorded at each time step.
    Notes

    """

    E = YoungsModulus(G, nu)
    L = R
    growth_factor = dt_growth

    
    C11, C12, C13, C14, C15, C16 = (1-nu), nu, nu, 0, 0, 0
    C21, C22, C23, C24, C25, C26 = nu, (1-nu), nu, 0, 0, 0
    C31, C32, C33, C34, C35, C36 = nu, nu, (1-nu), 0, 0, 0
    C41, C42, C43, C44, C45, C46 = 0, 0, 0, (1-2*nu), 0, 0
    C51, C52, C53, C54, C55, C56 = 0, 0, 0, 0, (1-2*nu), 0
    C61, C62, C63, C64, C65, C66 = 0, 0, 0, 0, 0, (1-2*nu)

    cauchy_values = (C11, C12, C13, C14, C15, C16,
                 C21, C22, C23, C24, C25, C26,
                 C31, C32, C33, C34, C35, C36,
                 C41, C42, C43, C44, C45, C46,
                 C51, C52, C53, C54, C55, C56,
                 C61, C62, C63, C64, C65, C66)

    Cauchy_tensor = (E)/((1+nu)*(1-2*nu))*CoefficientFunction(cauchy_values, dims=(6, 6)).Compile()

    Cauchy_tensor_star = (1/((1+nu)*(1-2*nu))) * CoefficientFunction(cauchy_values, dims=(6, 6)).Compile()



    def VoigtStrain(u):
        eps = Sym(Grad(u))
        # Standard Voigt order: 11, 22, 33, 23, 13, 12
        return CF( (eps[0,0], eps[1,1], eps[2,2], 
                    2*eps[1,2], 2*eps[0,2], 2*eps[0,1]) )


    def Stress_star_Anisotropic(u_vec):

        eps_v = VoigtStrain(u_vec)
        
        sigma_v = Cauchy_tensor_star * eps_v
        
        return CF( (sigma_v[0], sigma_v[5], sigma_v[4],
                    sigma_v[5], sigma_v[1], sigma_v[3],
                    sigma_v[4], sigma_v[3], sigma_v[2]), dims=(3,3) )
    k_ref = k

    # Parameter defining
    K = BulkModulus(G, nu) # [Pa] Bulk modulus
    S = n * chi + ((1-alpha)*(alpha-n)) / K # Storativity of the material
    E = YoungsModulus(G, nu) # [Pa] Young's modulus
    tau = L**2 * viscosity * S / k_ref # [s] Poroelastic time scale
    dt_star = dt / tau # Non-dimensional time step size
   
    
    S_star = S * E
    C = alpha / S_star # Coupling coefficient

    
    kxx, kxy, kxz = k_ref, 0.0, 0.0
    kyx, kyy, kyz = 0.0, k_ref, 0
    kzx, kzy, kzz = 0.0, 0.0, k_ref

    k_values = (kxx/k_ref, kxy/k_ref, kxz/k_ref,
                kyx/k_ref, kyy/k_ref, kyz/k_ref,
                kzx/k_ref, kzy/k_ref, kzz/k_ref)

    k_star = CoefficientFunction(k_values, dims=(3, 3)).Compile()
        
    g = 9.81

    R_Star = R / H
    H_Star = 1.0

    # Geometry and mesh generation
    cyl = Cylinder((0, 0, 0), Y, R_Star, H_Star)
    cyl.faces.name = "sides"
    cyl.faces.Max(Y).name = "top"
    cyl.faces.Min(Y).name = "bottom"

    mesh = Mesh(OCCGeometry(cyl).GenerateMesh(maxh=h)).Curve(3)

    # Finite element spaces

    V = VectorH1(mesh, order=order, dirichlet="top|bottom", dirichletx="sides", dirichletz="sides")
    Q = H1(mesh,order=order-1, dirichlet="top")

    (u,v) = V.TnT()
    (p,q) = Q.TnT()

    gfu_star = GridFunction(V)
    gfp_star = GridFunction(Q)
    gfp_star.Set(0)

    u_old_star = GridFunction(V)
    u_old_star.Set((0,0,0))

    p_old_star = GridFunction(Q)
    p_old_star.Set(0)

    # Weak forms
    traction = 0 # Traction BC

    tbar = CoefficientFunction((0, traction, 0))
    b = CoefficientFunction((0, -rho*g , 0)) # Body force

    qbar = CoefficientFunction(0) # Flux BC

    # K* (Stiffness) - Dimensionless
    a_K = BilinearForm(V)
    a_K += InnerProduct(Cauchy_tensor_star * VoigtStrain(u), VoigtStrain(v)) * dx
    pre_a_K = Preconditioner(a_K, type="bddc")
    a_K.Assemble()

    # Q* (Coupling) - Note: alpha is often kept here or used as a multiplier
    # Q* corresponds to the integral of div(u)*p
    a_Q12 = BilinearForm(trialspace=Q,testspace=V)
    a_Q12 += p * div(v) * dx
    a_Q12.Assemble()

    a_Q21 = BilinearForm(trialspace=V,testspace=Q)
    a_Q21 += q * div(u) * dx
    a_Q21.Assemble()

    # S* (Storage) - Pure L2 mass matrix
    a_S = BilinearForm(Q)
    a_S += p * q * dx
    pre_a_S = Preconditioner(a_S, type="local")
    a_S.Assemble()

    # H* (Conductivity) - Pure Laplacian
    a_H = BilinearForm(Q)
    a_H += InnerProduct(k_star * Grad(p), Grad(q)) * dx
    pre_a_H = Preconditioner(a_H, type="local")
    a_H.Assemble()

    # Preconditioner for Displacement
    a_PreP = BilinearForm(Q)
    a_PreP += (Grad(p) * Grad(q) + p * q) * dx
    pre_a_P = Preconditioner(a_PreP, type="bddc") # Use AMG for the pressure too
    a_PreP.Assemble()

    # Normalize traction and body forces
    tbar_star = tbar / E
    b_star = (b * L) / E

    b_f = LinearForm(V)
    b_f += (v * tbar_star) * ds("top") + (v * b_star) * dx
    b_f.Assemble()

    # qbar usually represents flux, scale appropriately
    qbar_star = (qbar * L) / (viscosity * E) 
    b_q = LinearForm(Q)
    b_q += qbar_star * q * ds("top")
    b_q.Assemble()

    # Time-stepping loop

    time_vals_nondim, F_solid, F_fluid = [], [], []
    t = 0

    t_end_star = t_end / tau
    u_max = 1e-4 # [m] Maximum displacement


    u_max_star = u_max / L
    t_ramp = 1 # [s] Time to reach full load
    t_ramp_star = t_ramp / tau


    # 1. Define the normal vector and vertical direction
    n = specialcf.normal(mesh.dim)

    dt_star_max = dt_max / tau # Set a reasonable upper limit for your physical scaled time step size

  
    A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q12.mat],
        [C * a_Q21.mat,   dt_star * a_H.mat + S_star * a_S.mat]
    ])

    F = BlockVector([
        b_f.vec, 
        C * a_Q21.mat * u_old_star.vec + S_star * a_S.mat * p_old_star.vec + dt_star * b_q.vec
    ])

    # Create the Block solution vector (linking to your GridFunctions)
    sol = BlockVector([gfu_star.vec, gfp_star.vec])
    # Pre-allocate memory for the residual and correction to avoid recreation
    rhs_resid = F.CreateVector()
    correction = F.CreateVector()

    # 1. Calculate the total vertical reaction force on the bottom
    # Create a mask for the bottom boundary nodes

    # This is for solid force
    v_test = GridFunction(V)
    v_test.Set(CF((0, 1, 0)), definedon=mesh.Boundaries("bottom"))
    # Before the loop, ensure u_old and p_old are initialized to zero (or initial state)

    while t < t_end_star:
        t += dt_star
        print(f"Time: {t*tau:.4f} s / {t_end:.4f} s", end='\r')
        uy_star = min(t / t_ramp_star, 1.0) * u_max_star
        disp_cf_star = CF((0, -uy_star, 0))
        gfu_star.Set(disp_cf_star, definedon=mesh.Boundaries("top"))
        # 1. Update the RHS (F) FIRST using the state from the END of the previous step
        # This ensures the 'source' for pressure is based on the previous equilibrium

        A = BlockMatrix([
        [a_K.mat,        -alpha * a_Q12.mat],
        [C * a_Q21.mat,   dt_star * a_H.mat + S_star * a_S.mat]
        ])
        #F[0].data = b_f.vec
        F[1].data = (C * a_Q21.mat * u_old_star.vec + 
                    S_star * a_S.mat * p_old_star.vec + 
                    dt_star * b_q.vec)

        pre_C = BlockMatrix([
        [pre_a_K, None],
        [None, dt_star * pre_a_P]
        ])
        

        # 3. Synchronize the 'sol' BlockVector
        sol[0].data = gfu_star.vec
        sol[1].data = gfp_star.vec

        # 4. Calculate the residual
        # Since sol[0] has the new BC and F[1] has the old state, 
        # the solver correctly identifies the 'change' over dt.
        rhs_resid.data = F - A * sol

        ################################
        fluid_force_star = Integrate(gfp_star, mesh, definedon=mesh.Boundaries("bottom"))


        fluid_force_phys = fluid_force_star * (E * L**2)

        F_fluid.append(fluid_force_phys)
        ################################
        
        rhs_resid[0].data[~V.FreeDofs()] = 0.0
        rhs_resid[1].data[~Q.FreeDofs()] = 0.0



        #print("Relative pressure residual:", Rp / bp if bp > 0 else 0)
        # 5. Solve and update
        correction[:] = 0.0
        GMRes(A=A, b=rhs_resid, pre=pre_C, x=correction, tol=1e-10, printrates=False, maxsteps=50)
        sol.data += correction


        sigma_star = Stress_star_Anisotropic(gfu_star)
        
        # Use the Virtual Work method to get the dimensionless force
        # Note: v_test only needs to be defined once outside the loop
        solid_force_star = Integrate(InnerProduct(sigma_star, Grad(v_test)), mesh)
        
        # Unscale and store
        solid_force_phys = solid_force_star * (E * L**2)
        F_solid.append(solid_force_phys)


        print(f"Solid: {solid_force_phys:.6f}, Fluid: {fluid_force_phys:.6f}, Total: {solid_force_phys + fluid_force_phys:.6f}, \
            Time: {t*tau:.4f} s / {t_end:.4f} s, dt: {dt_star * tau:.6f} s")

        #
        # 6. Handover (Update history for next step)
        u_old_star.vec.data = gfu_star.vec
        p_old_star.vec.data = gfp_star.vec

        time_vals_nondim.append(t)

        
   
        dt_star = min(dt_star * growth_factor, dt_star_max)



    time_vals = np.array(time_vals_nondim)*tau
    F_solid = np.array(F_solid)
    F_fluid = np.array(F_fluid)

    time_vals = np.insert(time_vals, 0, 0.0)
    F_solid =  np.insert(F_solid, 0, 0.0)
    F_fluid =  np.insert(F_fluid, 0, 0.0)

    return time_vals, F_solid, F_fluid

