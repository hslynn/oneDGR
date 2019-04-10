from dolfin import *
def project_functions(func_forms, func_list):
    for idx in range(len(func_forms)):
        func_space = func_list[idx].function_space() 
        u = TrialFunction(func_space)
        v = TestFunction(func_space)
        F = u*v*dx - func_forms[idx]*v*dx
        a, L = lhs(F), rhs(F)
        solve(a == L, func_list[idx])

def get_invg_forms(var_list):
    g00, g01, g11 = var_list[:3]
    invg_forms = []
    invg_forms.append(g11/(g00*g11-g01*g01))
    invg_forms.append(g01/(g00*g11-g01*g01))
    invg_forms.append(g00/(g00*g11-g01*g01))

    tuple(invg_forms)
    return invg_forms

def get_auxi_forms(var_list, invg_list):
    lapse = 1./pow(-invg_list[0], 0.5)
    shift = -invg_list[1]/invg_list[2]
    gamma11 =  1/var_list[2] 
    normal0 = 1/lapse
    normal1 = -shift/lapse
    return (lapse, shift, normal0, normal1, gamma11)

def get_T_forms(var_list, invg_list, auxi_list):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]
    
    T_scalar = Pi_psi*Pi_psi - gamma11*Phi_psi*Phi_psi
    T00 = gamma11*gamma11*g01*g01*Phi_psi*Phi_psi + 2*gamma11*g01*(-lapse)*Phi_psi*Pi_psi
             + lapse*lapse*Pi_psi*Pi_psi + 0.5*g00*T_scalar
    T01 = gamma11*gamma11*g01*g11*Phi_psi*Phi_psi + gamma11*g11*(-lapse)*Phi_psi*Pi_psi + 0.5*g01*T_scalar
    T11 = gamma11*gamma11*g11*g11*Phi_psi*Phi_psi + 0.5*g11*T_scalar
    return (T00, T01, T11, T_scalar)
    

def get_gamma_forms(var_list, auxi_list):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]

    gamma000 = 1/*0.5*(2*g01*Phi00) - 0.5*gamma11*g01*Phi00 + lapse*Pi00 - 1/2*lapse*Pi00
    gamma001 = gamma11*0.5*(g01*Phi01+g11*Phi00) - gamma11*1/2*g01*Phi01
    gamma011 = gamma11*0.5*(2*g11*Phi01) - gamma11*g01*Phi11 + 0.5*lapse*Pi11

    gamma100 = gamma11*0.5*(2*g01*Phi01) - 0.5*Phi00 + lapse*Pi01
    gamma101 = gamma11*0.5*(g01*Phi11 + g11*Phi01) - 0.5*Phi01 + 0.5*lapse*Pi11
    gamma111 = gamma11*0.5*(2*g11*Phi11) - 0.5*Phi11
    return (gamma000, gamma001, gamma011, gamma100, gamma101, gamma111)
    
def get_source_forms(var_list, invg_list, gamma_list, auxi_list, r):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    S, Pi_S, Phi_S = var_list[9:12]
    psi, Pi_psi, Phi_psi = var_list[12:15]

    gamma000, gamma001, gamma011, gamma100, gamma101, gamma111 = gamma_list[:]
    
    #source terms for g_AB
    src_g00 = -lapse*Pi00 - paragamma1*shift*Phi00
    src_g01 = -lapse*Pi01 - paragamma1*shift*Phi01
    src_g11 = -lapse*Pi11 - paragamma1*shift*Phi11

    #source terms for Pi_AB
    src_Pi00 = 2*lapse*(invg00(gamma11*Phi00*Phi00-Pi00*Pi00-invg00*gamma000*gamma000
            -2*invg01*gamma000*gamma001 - invg11*gamma001*gamma001)
            + invg01*(gamma11*Phi00*Phi01-Pi00*Pi01-invg00*gamma000*gamma001
            - invg01*(gamma000*gamma011+gamma001*gamma001)-invg11*gamma001*gamma011)
            + invg01*(gamma11*Phi00*Phi01-Pi00*Pi01-invg00*gamma001*gamma000
            - invg01*(gamma001*gamma001+gamma011*gamma000)-invg11*gamma011*gamma001)
            + invg11*(gamma11*Phi01*Phi01-Pi01*Pi01-invg00*gamma001*gamma001
            - invg01*2*gamma001*gamma011-invg11*gamma011*gamm011)) #term 1
            - 0.5*lapse*Pi00*(normal0*normal0*Pi00+normal0*normal1*2*Pi01+normal1*normal1*Pi11) #term 2
            - lapse*gamma11*Phi00(normal0*Pi01+normal1*Pi11)  #term3

    

    #source terms for Phi_AB 
    src_Phi00 = lapse*(0.5*Pi00*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)
            + gamma11*Phi00(normal0*Phi01 + normal1*Phi11) - gamma2*Phi00)
    src_Phi01 = lapse*(0.5*Pi01*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)
            + gamma11*Phi01(normal0*Phi01 + normal1*Phi11) - gamma2*Phi01)
    src_Phi00 = lapse*(0.5*Pi11*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)
            + gamma11*Phi11(normal0*Phi01 + normal1*Phi11) - gamma2*Phi11)

    #source terms for S
    src_S = -lapse*Pi_S - paragamma1*shift*Phi_S
    src_Pi_S = -lapse*gamma11*(normal0*Pi01 + normal1*Pi11) - 0.5*lapse*Pi_S*(normal0*normal0*Pi00 
            + 2*normal0*normal1*Pi01 + normal1*normal1*Pi11) - paragamma1*paragamma2*shift*Phi_S #row 1
            - 2*lapse/r/r*invg00*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))
            - 4*lapse/r/r*invg00*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*gamma11*g11*(r*Phi_S+1)
            - 2*lapse/r/r*invg11*(gamma11*g11*(r*Phi_S+1))*(gamma11*g11*(r*Phi_S+1)) #row 2
            - lapse/r*invg00*H0*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))
            - lapse/r*invg01*H0*gamma11*g11*(r*Phi_S+1)-lapse/r*invg01*H1*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))
            - lapse/r*invg11*H1*gamma11*g11*(r*Phi_S+1) #row 3
            - 2*lapse*(Pi_S*Pi_S-gamma11*Phi_S*Phi_S) + 4*lapse/r*(normal1*Pi_S-gamma11*Phi_S)
            + 3/r/r*(lapse*gamma11+shift*normal1) + 8/r*lapse*Phi_S + lapse/r/r*exp(-2*S) #row4
    src_Phi_S = lapse*(0.5*Pi_S*(normal0*normal*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)
            + gamma11*Pi_S*(normal0*Phi01 + normal1*Phi11) - gamma2*Phi_S)
    return [] 

    #source terms for psi
    src_psi = -lapse*Pi_psi - paragamma1*shift*Phi_psi
    src_Pi_psi = lapse*gamma11*gamma1*Phi_psi + lapse*Pi_psi*(invg00*gamma0*(-lapse)+invg01*gamma1*(-lapse))
            - paragamma1*paragamma2*shift*Phi_psi - lapse*(gamma11*Phi_psi*(normal0*Pi01+normal1*Pi11)
            + 0.5*Pi_psi*(normal0*normal0*Pi00 + 2*normal0*normal1*Pi01 + normal1*normal1*Pi11))
    src_Phi_psi = lapse*(0.5*Pi_psi*(normal0*normal*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)
           + gamma11*Phi_psi*(normal0*Phi01 + normal1*Phi11) - gamma2*Phi_psi)

def get_Hhat_forms(var_list, deri_list, invg_list, lapse, shift):
    return []

def get_deri(p_solution, u, i, mark):
    func_space = u.function_space() 
    mesh = func_space.mesh()
    p = TrialFunction(func_space)
    v = TestFunction(func_space)
    n = FacetNormal(mesh)

    term_cell = p*v*dx + u*v.dx(i)*dx
    if mark == '+':
        term_facet = - n("+")[i]*avg(u)*jump(v)*dS - n[i]*u*v*ds - 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
    elif mark == '-':
        term_facet = - n("+")[i]*avg(u)*jump(v)*dS - n[i]*u*v*ds + 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
    
    F = term_cell + term_facet
    a, L = lhs(F), rhs(F)
    solve(a == L, p_solution)

