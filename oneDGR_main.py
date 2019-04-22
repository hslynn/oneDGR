from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
from hdw import *
import rk2
import sys
import getopt

def main():
    """
    main computating process
    """
    DG_degree = 2
    mesh_num = 10

    opts, dumps = getopt.getopt(sys.argv[1:], "-m:-d:")
    for opt, arg in opts:
        if opt == "-m":
            mesh_num = int(arg)
        if opt == "-d":
            DG_degree = int(arg)

    #create mesh and define function space
    mesh = IntervalMesh(mesh_num, 1, 2)
    #mesh = UnitSquareMesh(mesh_num, mesh_num)
    #mesh = UnitCubeMesh(mesh_num, mesh_num, mesh_num)
    func_space = FunctionSpace(mesh, "DG", DG_degree)

    r = SpatialCoordinate(mesh)[0]

    #define functions for the variables
    var_list = [Function(func_space) for dummy in range(15)]
    #define functions for the auxi variables
    invg_list = [Function(func_space) for dummy in range(3)]
    auxi_list = [Function(func_space) for dummy in range(5)]
    gamma_list = [Function(func_space) for dummy in range(8)]
    T_list = [Function(func_space) for dummy in range(4)]
    C_list = [Function(func_space) for dummy in range(2)]
    H_list = [Function(func_space) for dummy in range(2)]
    deriH_list = [Function(func_space) for dummy in range(4)]


    Hhat_list = [Function(func_space) for dummy in range(15)]
    src_list = [Function(func_space) for dummy in range(15)]
    rhs_list = [Function(func_space) for dummy in range(15)]
    
    #initialize variables
    u_n = Expression("2*x[0]", degree=10)
    zero_exp = Expression("0", degree=10)

    #create functions for derivative of variables
    deri_list = []
    for idx in range(len(var_list)):
        var_list[idx] = project(zero_exp, func_space)
        deri_list.append([Function(func_space), Function(func_space)])

    var_list[0] = project(Expression("x[0] + 0.5", degree = 10), func_space)
    var_list[1] = project(Expression("0", degree = 10), func_space)
    var_list[2] = project(Expression("1/(x[0]+0.5)", degree = 10), func_space)

    C_list = [project(zero_func, func_space) for dummy in range(2)]
    H_list = [project(zero_func, func_space) for dummy in range(2)]
    deriH_list = [project(zero_func, func_space) for dummy in range(4)]

    #compute the derivative of variables
    for idx in range(len(var_list)):
        get_deri(deri_list[idx][0], var_list[idx], 0, "+")
        get_deri(deri_list[idx][1], var_list[idx], 0, "-")

    #temp
    avg_deri_list = [deri[0] for deri in deri_list]
    dif_deri_list = []
    
    #create form for middle terms
    invg_forms = get_invg_forms(var_list)
    auxi_forms = get_auxi_forms(var_list, invg_list) 
    gamma_forms = get_gamma_forms(var_list, invg_list, auxi_list, r)
    T_forms = get_T_forms(var_list, invg_list, auxi_list)

    Hhat_forms = get_Hhat_forms(var_list, avg_deri_list, dif_deri_list, auxi_list)
    src_forms = get_source_forms(var_list, invg_list, gamma_list, auxi_list, T_list, C_list, H_list, deriH_list, r)
    # rhs_forms = get_rhs_forms(Hhat_forms, src_forms)

    #Runge Kutta step
    project_functions(invg_forms, invg_list) 
    project_functions(auxi_forms, auxi_list)       
    project_functions(gamma_forms, gamma_list) 
    project_functions(T_forms, T_list)

    project_functions(src_forms, src_list)
    project_functions(Hhat_forms, Hhat_list)
    #project_functions(rhs_forms, rhs_list)
    
    #for vert in vertices(mesh):
    #    idx = vert.index() 
    #    print(str(mesh.coordinates()[idx]) + "=" + str(p.compute_vertex_values()[idx]))
    
    #error_L2 = errornorm(u_x, p, 'L2')
    #print("L2 error of p_+  = ", error_L2)
    
    plot(invg_list[2])
    plot(invg_list[0])
    plot(mesh)
    plt.show()

main()
