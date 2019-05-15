from dolfin import *
import hdw

def rk2(var_list, boundary_list, temp_var_list, deri_list, form_packs, func_packs, dt, t_now):
    mesh = var_list[0].function_space().mesh()
    for dummy in range(2): 
        for idx in range(len(var_list)):
            hdw.get_deri(deri_list[idx][0], var_list[idx], boundary_list[idx], 0, "+")
            hdw.get_deri(deri_list[idx][1], var_list[idx], boundary_list[idx], 0, "-")
        for idx in range(len(form_packs)):
            hdw.project_functions(form_packs[idx], func_packs[idx])
            for func_idx in range(len(func_packs[idx])):
                for vert in vertices(mesh):                                                                           
                    vert_idx = vert.index()                                                                                   
                    value = func_packs[idx][func_idx].compute_vertex_values()[vert_idx]
        rhs_list = func_packs[-1]
               
        dt_forms = [var_list[idx] + dt*rhs_list[idx] for idx in range(len(var_list))]
        hdw.project_functions(dt_forms, var_list)

        for vert in vertices(mesh):                                                                           
            var = var_list[0]
            temp_var = temp_var_list[0]
            rhs = rhs_list[0]
            vert_idx = vert.index()
    final_forms = [0.5*(temp_var_list[idx] + var_list[idx]) for idx in range(len(var_list))]
    hdw.project_functions(final_forms, var_list) 
