"""
help functions
"""
import numpy as np
from dolfin import *
from global_def import *

def get_deri(u_deri, u, bdry_values, i, mark):
    func_space = u.function_space() 
    mesh = func_space.mesh()
    p = TrialFunction(func_space)
    v = TestFunction(func_space)
    n = FacetNormal(mesh)

    term_cell = p*v*dx + u*v.dx(i)*dx

    left_bdry, right_bdry = [Constant(value) for value in bdry_values] 
    if mark == '+':
        term_inner_facet = - n("+")[i]*avg(u)*jump(v)*dS + 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
        term_boundary = -(0.5*(right_bdry + u)*n[i] + 0.5*(right_bdry - u)*abs(n[i]))*v*ds
    elif mark == '-':
        term_inner_facet = - n("+")[i]*avg(u)*jump(v)*dS - 0.5*abs(n("+")[i])*jump(u)*jump(v)*dS
        term_boundary = -(0.5*(left_bdry + u)*n[i] - 0.5*(left_bdry - u)*abs(n[i]))*v*ds
        
    F = term_cell + term_inner_facet + term_boundary
    a, L = lhs(F), rhs(F)
    solve(a == L, u_deri)


def project_functions(func_forms, func_list):
   for idx in range(len(func_forms)):
       project(func_forms[idx], func_list[idx].function_space(), function=func_list[idx])

def get_invg_forms(var_list):
    g00, g01, g11 = var_list[:3]
    invg_forms = []
    invg_forms.append(g11/(g00*g11-g01*g01))
    invg_forms.append(-g01/(g00*g11-g01*g01))
    invg_forms.append(g00/(g00*g11-g01*g01))

    tuple(invg_forms)
    return invg_forms

def get_auxi_forms(var_list, invg_list):
    lapse = 1./pow(-invg_list[0], 0.5)
    shift = -invg_list[1]/invg_list[0]
    gamma11 =  1/var_list[2] 
    normal0 = 1/lapse
    normal1 = -shift/lapse
    return (lapse, shift, normal0, normal1, gamma11)

def find_AH(var_list, auxi_list, left_bdry, right_bdry, step_len):
    g11 = var_list[2]
    lapse = auxi_list[0]
    shift = auxi_list[1] 
    AH_indicator_left = 1/g11(left_bdry)**0.5 - shift(left_bdry)/lapse(left_bdry) 
    for coord in np.arange(left_bdry + step_len, right_bdry, step_len):
        AH_indicator = 1/g11(coord)**0.5 - shift(coord)/lapse(coord) 
        if AH_indicator*AH_indicator_left < 0:
            return (coord - step_len, coord)
        elif AH_indicator*AH_indicator_left == 0:
            return (coord, coord)
        else:
            AH_indicator_left = AH_indicator

    return "no apparent horizon has been found between " + str((left_bdry, right_bdry)) 



def get_T_forms(var_list, invg_list, auxi_list):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    S, Pi_S, Phi_S = var_list[9:12]
    psi, Pi_psi, Phi_psi = var_list[12:15]
    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]
    
    T_scalar = Pi_psi*Pi_psi - gamma11*Phi_psi*Phi_psi
    T00 = gamma11*gamma11*g01*g01*Phi_psi*Phi_psi + 2*gamma11*g01*(-lapse)*Phi_psi*Pi_psi \
             + lapse*lapse*Pi_psi*Pi_psi + 0.5*g00*T_scalar
    T01 = gamma11*gamma11*g01*g11*Phi_psi*Phi_psi + gamma11*g11*(-lapse)*Phi_psi*Pi_psi + 0.5*g01*T_scalar
    T11 = gamma11*gamma11*g11*g11*Phi_psi*Phi_psi + 0.5*g11*T_scalar
    return (T00, T01, T11, T_scalar)
    

def get_gamma_forms(var_list, invg_list, auxi_list, r):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    S, Pi_S, Phi_S = var_list[9:12]

    invg00, invg01, invg11 = invg_list[:]
    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]

    gamma000 = gamma11*0.5*(2*g01*Phi00) - 0.5*gamma11*g01*Phi00 - lapse*Pi00 + 1/2*lapse*Pi00
    gamma001 = gamma11*0.5*(g01*Phi01+g11*Phi00) - gamma11*1/2*g01*Phi01
    gamma011 = gamma11*0.5*(2*g11*Phi01) - 0.5*gamma11*g01*Phi11 + 0.5*lapse*Pi11

    gamma100 = gamma11*0.5*(2*g01*Phi01) - 0.5*Phi00 - lapse*Pi01
    gamma101 = gamma11*0.5*(g01*Phi11 + g11*Phi01) - 0.5*Phi01 - 0.5*lapse*Pi11
    gamma111 = gamma11*0.5*(2*g11*Phi11) - 0.5*Phi11

    gamma0 = invg00*gamma000 + 2*invg01*gamma001 + invg11*gamma011 \
            - 2/r*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))
    gamma1 = invg00*gamma100 + 2*invg01*gamma101 + invg11*gamma111 \
            - 2/r*(gamma11*g11*(r*Phi_S+1))

    return (gamma000, gamma001, gamma011, gamma100, gamma101, gamma111, gamma0, gamma1)

def get_C_forms(H_list, gamma_list):
    H0, H1 = H_list[:]
    gamma0, gamma1 = gamma_list[6:]
    return (H0+gamma0, H1+gamma1)
    
def get_source_forms(var_list, invg_list, auxi_list, gamma_list, T_list, C_list, H_list, deriH_list, r):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    S, Pi_S, Phi_S = var_list[9:12]
    psi, Pi_psi, Phi_psi = var_list[12:15]

    invg00, invg01, invg11 = invg_list[:]
    gamma000, gamma001, gamma011, gamma100, gamma101, gamma111, gamma0, gamma1 = gamma_list[:]
    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]
    T00, T01, T11, T_scalar = T_list[:]
    C0, C1 = C_list[:]
    H0, H1 = H_list[:]
    deriH00, deriH01, deriH10, deriH11 = deriH_list[:]
    
    #source terms for g_AB
    src_g00 = -lapse*Pi00 - paragamma1*shift*Phi00
    src_g01 = -lapse*Pi01 - paragamma1*shift*Phi01
    src_g11 = -lapse*Pi11 - paragamma1*shift*Phi11

    #source terms for Pi_AB
    src_Pi00 = (2*lapse*(invg00*(gamma11*Phi00*Phi00-Pi00*Pi00-invg00*gamma000*gamma000 \
            -2*invg01*gamma000*gamma001 - invg11*gamma001*gamma001) \
            + invg01*(gamma11*Phi00*Phi01-Pi00*Pi01-invg00*gamma000*gamma001 \
            - invg01*(gamma000*gamma011+gamma001*gamma001)-invg11*gamma001*gamma011) \
            + invg01*(gamma11*Phi00*Phi01-Pi00*Pi01-invg00*gamma001*gamma000 \
            - invg01*(gamma001*gamma001+gamma011*gamma000)-invg11*gamma011*gamma001) \
            + invg11*(gamma11*Phi01*Phi01-Pi01*Pi01-invg00*gamma001*gamma001 \
            - invg01*2*gamma001*gamma011-invg11*gamma011*gamma011)) \
            #term 1
            - 0.5*lapse*Pi00*(normal0*normal0*Pi00+2*normal0*normal1*Pi01+normal1*normal1*Pi11) \
            #term 2
            - lapse*gamma11*Phi00*(normal0*Pi01+normal1*Pi11) \
            #term 3
            - 2*lapse*(deriH00 + invg00*gamma000*(paragamma4*C0-H0) + invg01*gamma000*(paragamma4*C1-H1) \
            + invg01*gamma100*(paragamma4*C0-H0) + invg11*gamma100*(paragamma4*C1-H1) \
            - 0.5*paragamma5*g00*(invg00*gamma0*C0+invg01*(gamma0*C1+gamma1*C0)+invg11*gamma1*C1)) \
            #term 4
            + lapse*paragamma0*((-2*lapse-g00*normal0)*C0+(-g00*normal1)*C1)  \
            #term 5
            - paragamma1*paragamma2*shift*Phi00 \
            #term 6
            - 4*lapse/pow(r,2)*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*(gamma11*g01*(r*Phi_S+1) \
            -lapse*(r*Pi_S-normal1)) \
            #term 7
            + 16*pi*lapse*(T00-0.5*T_scalar*g00))

    src_Pi01 = (2*lapse*(invg00*(gamma11*Phi00*Phi01-Pi00*Pi01-invg00*gamma000*gamma100 \
            - invg01*(gamma000*gamma101+gamma001*gamma100) - invg11*gamma001*gamma101) \
            + invg01*(gamma11*Phi00*Phi11-Pi00*Pi11-invg00*gamma000*gamma101 \
            - invg01*(gamma000*gamma111+gamma001*gamma101)-invg11*gamma001*gamma111) \
            + invg01*(gamma11*Phi01*Phi01-Pi01*Pi01-invg00*gamma001*gamma100 \
            - invg01*(gamma001*gamma101+gamma011*gamma100)-invg11*gamma011*gamma101) \
            + invg11*(gamma11*Phi01*Phi11-Pi01*Pi11-invg00*gamma001*gamma101 \
            - invg01*(gamma001*gamma111+gamma011*gamma101)-invg11*gamma011*gamma111)) \
            #term 1
            - 0.5*lapse*Pi01*(normal0*normal0*Pi00+normal0*normal1*2*Pi01+normal1*normal1*Pi11) \
            #term 2
            - lapse*gamma11*Phi01*(normal0*Pi01+normal1*Pi11)  \
            #term 3
            - 2*lapse*(0.5*(deriH01+deriH10)+invg00*gamma001*(paragamma4*C0-H0) \
            + invg01*gamma001*(paragamma4*C1-H1) \
            + invg01*gamma101*(paragamma4*C0-H0) + invg11*gamma101*(paragamma4*C1-H1) \
            - 0.5*paragamma5*g01*(invg00*gamma0*C0+invg01*(gamma0*C1+gamma1*C0)+invg11*gamma1*C1)) \
            #term 4
            + lapse*paragamma0*((-g01*normal0)*C0+(-lapse-g01*normal1)*C1) \
            #term 5
            - paragamma1*paragamma2*shift*Phi01 \
            #term 6
            - 4*lapse/pow(r,2)*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*(gamma11*g11*(r*Phi_S+1)) \
            #term 7
            + 16*pi*lapse*(T01-0.5*T_scalar*g01))

    src_Pi11 = (2*lapse*(invg00*(gamma11*Phi01*Phi01-Pi01*Pi01-invg00*gamma100*gamma100 \
            -2*invg01*gamma100*gamma101 - invg11*gamma101*gamma101) \
            + invg01*(gamma11*Phi01*Phi11-Pi01*Pi11-invg00*gamma100*gamma101 \
            - invg01*(gamma100*gamma111+gamma101*gamma101)-invg11*gamma101*gamma111) \
            + invg01*(gamma11*Phi11*Phi01-Pi11*Pi01-invg00*gamma101*gamma100 \
            - invg01*(gamma101*gamma101+gamma111*gamma100)-invg11*gamma111*gamma101) \
            + invg11*(gamma11*Phi11*Phi11-Pi11*Pi11-invg00*gamma101*gamma101 \
            - invg01*2*gamma101*gamma111-invg11*gamma111*gamma111)) \
            #term 1
            - 0.5*lapse*Pi11*(normal0*normal0*Pi00+normal0*normal1*2*Pi01+normal1*normal1*Pi11) \
            #term 2
            - lapse*gamma11*Phi11*(normal0*Pi01+normal1*Pi11) \
            #term 3
            - 2*lapse*(deriH11 + invg00*gamma011*(paragamma4*C0-H0) + invg01*gamma011*(paragamma4*C1-H1) \
            + invg01*gamma111*(paragamma4*C0-H0) + invg11*gamma111*(paragamma4*C1-H1) \
            - 0.5*paragamma5*g11*(invg00*gamma0*C0+invg01*(gamma0*C1+gamma1*C0)+invg11*gamma1*C1)) \
            #term 4
            + lapse*paragamma0*((-g11*normal0)*C0+(-g11*normal1)*C1) \
            #term 5
            - paragamma1*paragamma2*shift*Phi11 \
            #term 6
            - 4*lapse/pow(r,2)*(gamma11*g11*(r*Phi_S+1))*(gamma11*g11*(r*Phi_S+1)) \
            #term 7
            + 16*pi*lapse*(T11-0.5*T_scalar*g11))

    

    #source terms for Phi_AB 
    src_Phi00 = lapse*(0.5*Pi00*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11) \
            + gamma11*Phi00*(normal0*Phi01 + normal1*Phi11) - paragamma2*Phi00)
    src_Phi01 = lapse*(0.5*Pi01*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11) \
            + gamma11*Phi01*(normal0*Phi01 + normal1*Phi11) - paragamma2*Phi01)
    src_Phi11 = lapse*(0.5*Pi11*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11) \
            + gamma11*Phi11*(normal0*Phi01 + normal1*Phi11) - paragamma2*Phi11)

    #source terms for S
    src_S = -lapse*Pi_S - paragamma1*shift*Phi_S
    src_Pi_S = (-lapse*Phi_S*gamma11*(normal0*Pi01 + normal1*Pi11) - 0.5*lapse*Pi_S*(normal0*normal0*Pi00 \
            + 2*normal0*normal1*Pi01 + normal1*normal1*Pi11) - paragamma1*paragamma2*shift*Phi_S \
            #row 1
            - 2*lapse/r/r*invg00*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*(gamma11*g01*(r*Phi_S+1) \
            - lapse*(r*Pi_S-normal1)) \
            - 4*lapse/r/r*invg01*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1))*gamma11*g11*(r*Phi_S+1) \
            - 2*lapse/r/r*invg11*(gamma11*g11*(r*Phi_S+1))*(gamma11*g11*(r*Phi_S+1)) \
            #row 2
            - lapse/r*invg00*H0*(gamma11*g01*(r*Phi_S+1)-lapse*(r*Pi_S-normal1)) \
            - lapse/r*invg01*H0*gamma11*g11*(r*Phi_S+1)-lapse/r*invg01*H1*(gamma11*g01*(r*Phi_S+1) \
            - lapse*(r*Pi_S-normal1)) \
            - lapse/r*invg11*H1*gamma11*g11*(r*Phi_S+1) \
            #row 3
            - 2*lapse*(Pi_S*Pi_S-gamma11*Phi_S*Phi_S) + 4*lapse/r*(normal1*Pi_S-gamma11*Phi_S) \
            + 3/r/r*(lapse*gamma11+shift*normal1) + 8/r*lapse*Phi_S + lapse/r/r*exp(-2*S))
    src_Phi_S = lapse*(0.5*Pi_S*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11) \
            + gamma11*Pi_S*(normal0*Phi01 + normal1*Phi11) - paragamma2*Phi_S)

    #source terms for psi
    src_psi = -lapse*Pi_psi - paragamma1*shift*Phi_psi
    src_Pi_psi = lapse*gamma11*gamma1*Phi_psi + lapse*Pi_psi*(invg00*gamma0*(-lapse)+invg01*gamma1*(-lapse))\
            - paragamma1*paragamma2*shift*Phi_psi - lapse*(gamma11*Phi_psi*(normal0*Pi01+normal1*Pi11) \
            + 0.5*Pi_psi*(normal0*normal0*Pi00 + 2*normal0*normal1*Pi01 + normal1*normal1*Pi11))
    src_Phi_psi = lapse*(0.5*Pi_psi*(normal0*normal0*Phi00 + 2*normal0*normal1*Phi01 + normal1*normal1*Phi11)\
           + gamma11*Phi_psi*(normal0*Phi01 + normal1*Phi11) - paragamma2*Phi_psi)

    return (src_g00, src_g01, src_g11,
            src_Pi00, src_Pi01, src_Pi11,
            src_Phi00, src_Phi01, src_Phi11,
            src_S, src_Pi_S, src_Phi_S, 
            src_psi, src_Pi_psi, src_Phi_psi)

def get_Hhat_forms(var_list, deri_list, auxi_list):
    g00, g01, g11 = var_list[:3]
    Pi00, Pi01, Pi11 = var_list[3:6]
    Phi00, Phi01, Phi11 = var_list[6:9]
    S, Pi_S, Phi_S = var_list[9:12]
    psi, Pi_psi, Phi_psi = var_list[12:15]
   
    avg_deri_list = [0.5*(deri[0]+deri[1]) for deri in deri_list] 
    dif_deri_list = [0.5*(deri[0]-deri[1]) for deri in deri_list] 

    avg_deri_g00, avg_deri_g01, avg_deri_g11 = avg_deri_list[:3]
    avg_deri_Pi00, avg_deri_Pi01, avg_deri_Pi11 = avg_deri_list[3:6]
    avg_deri_Phi00, avg_deri_Phi01, avg_deri_Phi11 = avg_deri_list[6:9]
    avg_deri_S, avg_deri_Pi_S, avg_deri_Phi_S = avg_deri_list[9:12]
    avg_deri_psi, avg_deri_Pi_psi, avg_deri_Phi_psi = avg_deri_list[12:15]

    dif_deri_g00, dif_deri_g01, dif_deri_g11 = dif_deri_list[:3]
    dif_deri_Pi00, dif_deri_Pi01, dif_deri_Pi11 = dif_deri_list[3:6]
    dif_deri_Phi00, dif_deri_Phi01, dif_deri_Phi11 = dif_deri_list[6:9]
    dif_deri_S, dif_deri_Pi_S, dif_deri_Phi_S = dif_deri_list[9:12]
    dif_deri_psi, dif_deri_Pi_psi, dif_deri_Phi_psi = dif_deri_list[12:15]

    lapse, shift, normal0, normal1, gamma11 = auxi_list[:]

    Hhat_g00 = -(1+paragamma1)*shift*avg_deri_g00 - dif_deri_g00
    Hhat_g01 = -(1+paragamma1)*shift*avg_deri_g01 - dif_deri_g01
    Hhat_g11 = -(1+paragamma1)*shift*avg_deri_g11 - dif_deri_g11

    Hhat_Pi00 = -paragamma1*paragamma2*shift*avg_deri_g00 - shift*avg_deri_Pi00 \
            + lapse*gamma11*avg_deri_Phi00 - dif_deri_Pi00
    Hhat_Pi01 = -paragamma1*paragamma2*shift*avg_deri_g01 - shift*avg_deri_Pi01 \
            + lapse*gamma11*avg_deri_Phi01 - dif_deri_Pi01
    Hhat_Pi11 = -paragamma1*paragamma2*shift*avg_deri_g11 - shift*avg_deri_Pi11 \
            + lapse*gamma11*avg_deri_Phi11 - dif_deri_Pi11

    Hhat_Phi00 = -paragamma2*lapse*avg_deri_g00 + lapse*avg_deri_Pi00 - shift*avg_deri_Phi00 \
            - dif_deri_Phi00
    Hhat_Phi01 = -paragamma2*lapse*avg_deri_g01 + lapse*avg_deri_Pi01 - shift*avg_deri_Phi01 \
            - dif_deri_Phi01
    Hhat_Phi11 = -paragamma2*lapse*avg_deri_g11 + lapse*avg_deri_Pi11 - shift*avg_deri_Phi11 \
            - dif_deri_Phi11

    Hhat_S = -(1+paragamma1)*shift*avg_deri_S - dif_deri_S
    Hhat_Pi_S = -paragamma1*paragamma2*shift*avg_deri_S - shift*avg_deri_Pi_S + lapse*gamma11*avg_deri_Phi_S\
            - dif_deri_Pi_S
    Hhat_Phi_S = -paragamma2*lapse*avg_deri_S + lapse*avg_deri_Pi_S - shift*avg_deri_Phi_S - dif_deri_Phi_S

    Hhat_psi = -(1+paragamma1)*shift*avg_deri_psi - dif_deri_psi
    Hhat_Pi_psi = -paragamma1*paragamma2*shift*avg_deri_psi - shift*avg_deri_Pi_psi \
            + lapse*gamma11*avg_deri_Phi_psi - dif_deri_Pi_psi
    Hhat_Phi_psi = -paragamma2*lapse*avg_deri_psi + lapse*avg_deri_Pi_psi - shift*avg_deri_Phi_psi \
            - dif_deri_Phi_psi

    return (Hhat_g00, Hhat_g01, Hhat_g11,
            Hhat_Pi00, Hhat_Pi01, Hhat_Pi11,
            Hhat_Phi00, Hhat_Phi01, Hhat_Phi11,
            Hhat_S, Hhat_Pi_S, Hhat_Phi_S, 
            Hhat_psi, Hhat_Pi_psi, Hhat_Phi_psi)

def get_rhs_forms(Hhat_forms, src_forms):
    return tuple([src_forms[idx]-Hhat_forms[idx] for idx in range(len(src_forms))])
