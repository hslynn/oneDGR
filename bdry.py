"""
handle boundary conditons
"""

import numpy as np
from dolfin import *
from global_def import *

def get_characteristic_speed(auxi_list):
    lapse = auxi_list[0]
    shift = auxi_list[1]
    gamma11 = auxi_list[4]
        
    #left bdry
    lapse_left = lapse.compute_vertex_values()[0]
    shift_left = shift.compute_vertex_values()[0]
    gamma11_left = gamma11.compute_vertex_values()[0]
    
    v0_left = -(1+paragamma1)*shift_left    
    v_minus_left = -shift_left + lapse_left*gamma11_left**0.5   
    v_plus_left = -shift_left - lapse_left*gamma11_left**0.5   

    #right bdry
    lapse_right = lapse.compute_vertex_values()[-1]
    shift_right = shift.compute_vertex_values()[-1]
    gamma11_right = gamma11.compute_vertex_values()[-1]
    
    v0_right = -(1+paragamma1)*shift_right    
    v_minus_right = -shift_right + lapse_right*gamma11_right**0.5   
    v_plus_right = -shift_right - lapse_right*gamma11_right**0.5   

    return ((v0_left, v_minus_left, v_plus_left),
            (v0_right, v_minus_right, v_plus_right))
    

def get_characteristic_field_values(var_list):
    #field values at left boundary
    left_var_values = [var.compute_vertex_values()[0] for var in var_list]
    g00_left, g01_left, g11_left = left_var_values[:3]
    Pi00_left, Pi01_left, Pi11_left = left_var_values[3:6]
    Phi00_left, Phi01_left, Phi11_left = left_var_values[6:9]
    S_left, Pi_S_left, Phi_S_left = left_var_values[9:12]
    psi_left, Pi_psi_left, Phi_psi_left = left_var_values[12:15]

    u0_00_left = g00_left
    u0_01_left = g01_left
    u0_11_left = g11_left
    u0_S_left = S_left
    u0_psi_left = psi_left
 
    u_minus_00_left = -paragamma2*g00_left + Pi00_left + 1/g11_left**0.5*Phi00_left
    u_minus_01_left = -paragamma2*g01_left + Pi01_left + 1/g11_left**0.5*Phi01_left
    u_minus_11_left = -paragamma2*g11_left + Pi11_left + 1/g11_left**0.5*Phi11_left
    u_minus_s_left = -paragamma2*s_left + pi_s_left + 1/g11_left**0.5*phi_s_left
    u_minus_psi_left = -paragamma2*psi_left + pi_psi_left + 1/g11_left**0.5*phi_psi_left

    u_plus_00_left = -paragamma2*g00_left + Pi00_left - 1/g11_left**0.5*Phi00_left
    u_plus_01_left = -paragamma2*g01_left + Pi01_left - 1/g11_left**0.5*Phi01_left
    u_plus_11_left = -paragamma2*g11_left + Pi11_left - 1/g11_left**0.5*Phi11_left
    u_plus_S_left = -paragamma2*S_left + Pi_S_left + 1/g11_left**0.5*Phi_S_left
    u_plus_psi_left = -paragamma2*psi_left + Pi_psi_left + 1/g11_left**0.5*Phi_psi_left

    #field values at right boundary
    right_var_values = [var.compute_vertex_values()[-1] for var in var_list]
    g00_right, g01_right, g11_right = right_var_values[:3]
    Pi00_right, Pi01_right, Pi11_right = right_var_values[3:6]
    Phi00_right, Phi01_right, Phi11_right = right_var_values[6:9]
    S_right, Pi_S_right, Phi_S_right = right_var_values[9:12]
    psi_right, Pi_psi_right, Phi_psi_right = right_var_values[12:15]

    u0_00_right = g00_right
    u0_01_right = g01_right
    u0_11_right = g11_right
    u0_S_right = S_right
    u0_psi_right = psi_right
 
    u_minus_00_right = -paragamma2*g00_right + Pi00_right + 1/g11_right**0.5*Phi00_right
    u_minus_01_right = -paragamma2*g01_right + Pi01_right + 1/g11_right**0.5*Phi01_right
    u_minus_11_right = -paragamma2*g11_right + Pi11_right + 1/g11_right**0.5*Phi11_right
    u_minus_s_right = -paragamma2*s_right + pi_s_right + 1/g11_right**0.5*phi_s_right
    u_minus_psi_right = -paragamma2*psi_right + pi_psi_right + 1/g11_right**0.5*phi_psi_right

    u_plus_00_right = -paragamma2*g00_right + Pi00_right - 1/g11_right**0.5*Phi00_right
    u_plus_01_right = -paragamma2*g01_right + Pi01_right - 1/g11_right**0.5*Phi01_right
    u_plus_11_right = -paragamma2*g11_right + Pi11_right - 1/g11_right**0.5*Phi11_right
    u_plus_S_right = -paragamma2*S_right + Pi_S_right + 1/g11_right**0.5*Phi_S_right
    u_plus_psi_right = -paragamma2*psi_right + Pi_psi_right + 1/g11_right**0.5*Phi_psi_right

    return ((u0_00_left, u0_01_left, u0_11_left, u0_S_left, u0_psi_left,
            u_minus_00_left, u_minus_01_left, u_minus_11_left, u_minus_S_left, u_minus_psi_left,
            u_plus_00_left, u_plus_01_left, u_plus_11_left, u_plus_S_left, u_plus_psi_left),
            (u0_00_right, u0_01_right, u0_11_right, u0_S_right, u0_psi_right,
            u_minus_00_right, u_minus_01_right, u_minus_11_right, u_minus_S_right, u_minus_psi_right,
            u_plus_00_right, u_plus_01_right, u_plus_11_right, u_plus_S_right, u_plus_psi_right))
    

def get_bdry_values(var_list, auxi_list,  exact_characteristic_field_values):
    real_characteristic_field_values = get_characteristic_field_values(var_list)
    real_characteristic_speed = get_characteristic_speed(auxi_list)

    #left boundary
    exact_field_values_left = exact_characteristic_field_values[0]
    u0_00_left, u0_01_left, u0_11_left, u0_S_left, u0_psi_left = exact_field_values_left[:5]
    u_minus_00_left, u_minus_01_left, u_minus_11_left, u_minus_S_left, u_minus_psi_left = exact_field_values_left[5:10]
    u_plus_00_left, u_plus_01_left, u_plus_11_left, u_plus_S_left, u_plus_psi_left = exact_field_values_left[10:15]

    #determine outgoing and vertical fields using values at the boundary, replacing the values above
    real_field_values_left = real_characteristic_field_values[0]
    real_characteristic_speed_left = real_characteristic_speed[0]

    v0_left, v_minus_left, v_plus_left = real_characteristic_speed_left[:]
    if v0_left <= 0: 
        u0_00_left, u0_01_left, u0_11_left, u0_S_left, u0_psi_left = real_field_values_left[:5]
    if v_minus_left <= 0: 
        u_minus_00_left, u_minus_01_left, u_minus_11_left, u_minus_S_left, u_minus_psi_left = real_field_values_left[5:10]
    if v_plus_left <= 0: 
        u_plus_00_left, u_plus_01_left, u_plus_11_left, u_plus_S_left, u_plus_psi_left = real_field_values_left[10:15]

    #compute left boundary values using characteristic fields
    g00_lbdry = u0_00_left
    g01_lbdry = u0_01_left
    g11_lbdry = u0_11_left
    
    Pi00_lbdry = 0.5*(u_minus_00_left + u_plus_00_left + 2*paragamma2*u0_00_left)
    Pi01_lbdry = 0.5*(u_minus_01_left + u_plus_01_left + 2*paragamma2*u0_01_left)
    Pi11_lbdry = 0.5*(u_minus_11_left + u_plus_11_left + 2*paragamma2*u0_11_left)

    Phi00_lbdry = 0.5*(u0_11_left**0.5)*(u_minus_00_left - u_plus_00_left)
    Phi01_lbdry = 0.5*(u0_11_left**0.5)*(u_minus_01_left - u_plus_01_left)
    Phi11_lbdry = 0.5*(u0_11_left**0.5)*(u_minus_11_left - u_plus_11_left)

    S_lbdry = u0_S_left
    Pi_S_lbdry = 0.5*(u_minus_S_left + u_plus_S_left + 2*paragamma2*u0_S_left)
    Phi_S_lbdry = 0.5*(u0_11_left**0.5)*(u_minus_S_left - u_plus_S_left)

    psi_lbdry = u0_psi_left
    Pi_psi_lbdry = 0.5*(u_minus_psi_left + u_plus_psi_left + 2*paragamma2*u0_psi_left)
    Phi_psi_lbdry = 0.5*(u0_11_left**0.5)*(u_minus_psi_left - u_plus_psi_left)
    
    left_bdry_values = [g00_lbdry, g01_lbdry, g11_lbdry,
            Pi00_lbdry, Pi01_lbdry, Pi11_lbdry,
            Phi00_lbdry, Phi01_lbdry, Phi11_lbdry,
            S_lbdry, Pi_S_lbdry, Phi_S_lbdry,
            psi_lbdry, Pi_psi_lbdry, Phi_psi_lbdry]

    ##########################################################
    #right boundary
    exact_field_values_right = exact_characteristic_field_values[1] 
    u0_00_right, u0_01_right, u0_11_right, u0_S_right, u0_psi_right = exact_field_values_right[:5]
    u_minus_00_right, u_minus_01_right, u_minus_11_right, u_minus_S_right, u_minus_psi_right = exact_field_values_right[5:10]
    u_plus_00_right, u_plus_01_right, u_plus_11_right, u_plus_S_right, u_plus_psi_right = exact_field_values_right[10:15]
   
    #determine outgoing and vertical fields using values at the boundary, replacing the values above
    real_field_values_right = real_characteristic_field_values[1]
    real_characteristic_speed_right = real_characteristic_speed[1]
        
    v0_right, v_minus_right, v_plus_right = real_characteristic_speed_right[:]
    if v0_right >= 0: 
        u0_00_right, u0_01_right, u0_11_right, u0_S_right, u0_psi_right = real_field_values_right[:5]
    if v_minus_right >= 0: 
        u_minus_00_right, u_minus_01_right, u_minus_11_right, u_minus_s_right, u_minus_psi_right = real_field_values_right[5:10]
    if v_plus_right >= 0: 
        u_plus_00_right, u_plus_01_right, u_plus_11_right, u_plus_S_right, u_plus_psi_right = real_field_values_right[10:15]

    #compute right boundary values using characteristic fields
    g00_rbdry = u0_00_right
    g01_rbdry = u0_01_right
    g11_rbdry = u0_11_right

    Pi00_rbdry = 0.5*(u_minus_00_right + u_plus_00_right + 2*paragamma2*u0_00_right)
    Pi01_rbdry = 0.5*(u_minus_01_right + u_plus_01_right + 2*paragamma2*u0_01_right)
    Pi11_rbdry = 0.5*(u_minus_11_right + u_plus_11_right + 2*paragamma2*u0_11_right)

    Phi00_rbdry = 0.5*(u0_11_right**0.5)*(u_minus_00_right - u_plus_00_right)
    Phi01_rbdry = 0.5*(u0_11_right**0.5)*(u_minus_01_right - u_plus_01_right)
    Phi11_rbdry = 0.5*(u0_11_right**0.5)*(u_minus_11_right - u_plus_11_right)

    S_rbdry = u0_S_right
    Pi_S_rbdry = 0.5*(u_minus_S_right + u_plus_S_right + 2*paragamma2*u0_S_right)
    Phi_S_rbdry = 0.5*(u0_11_right**0.5)*(u_minus_S_right - u_plus_S_right)

    psi_rbdry = u0_psi_right
    Pi_psi_rbdry = 0.5*(u_minus_psi_right + u_plus_psi_right + 2*paragamma2*u0_psi_right)
    Phi_psi_rbdry = 0.5*(u0_11_right**0.5)*(u_minus_psi_right - u_plus_psi_right)


    right_bdry_values = [g00_rbdry, g01_rbdry, g11_rbdry,
            Pi00_rbdry, Pi01_rbdry, Pi11_rbdry,
            Phi00_rbdry, Phi01_rbdry, Phi11_rbdry,
            S_rbdry, Pi_S_rbdry, Phi_S_rbdry,
            psi_rbdry, Pi_psi_rbdry, Phi_psi_rbdry]

    return [(left_bdry_values[idx], right_bdry_values[idx]) for idx in range(len(var_list))]


