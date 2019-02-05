from dolfin import *
def projcet_functions(func_forms, func_list):
    for idx in range(len(func_forms)):
        func_space = func_list[idx].function_space() 
        project(func_forms[idx], func_space)

def get_invg_forms(var_list):
    g_00, g_01, g_11 = var_list[:3]
    invg_forms = []
    invg_forms.append(g_11/(g_00*g_11-g_01*g_01))
    invg_forms.append(g_01/(g_00*g_11-g_01*g_01))
    invg_forms.append(g_00/(g_00*g_11-g_01*g_01))
    return invg_forms
    
def get_source_forms(var_list, invg_list, lapse, shift):
    return [] 

def get_Hhat_forms(var_list, deri_list, invg_list, lapse, shift)
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
