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
            print(arg)
            mesh_num = int(arg)
        if opt == "-d":
            DG_degree = int(arg)

    #create mesh and define function space
    mesh = UnitIntervalMesh(mesh_num)
    #mesh = UnitSquareMesh(mesh_num, mesh_num)
    #mesh = UnitCubeMesh(mesh_num, mesh_num, mesh_num)
    func_space = FunctionSpace(mesh, "DG", DG_degree)

    #define functions for the variables
    var_list = [Function(func_space) for dummy in range(16)]
    #define functions for the auxi variables
    invg_list = [Function(func_space) for dummy in range(3)]
    lapse = Function(func_space)
    shift = Function(func_space)
    
    #initialize variables
    u_n = Expression("2*x[0]", degree=10)
    u_x = Expression("4", degree=10)

    #create functions for derivative of variables
    deri_list = []
    for idx in range(len(var_list)):
        var_list[idx] = project(u_n, func_space)
        deri_list.append([Function(func_space), Function(func_space)])

    var_list[0] = project(Expression("x[0] + 0.5", degree = 10), func_space)
    var_list[1] = project(Expression("0", degree = 10), func_space)
    var_list[2] = project(Expression("1/(x[0]+0.5)", degree = 10), func_space)

    #compute the derivative of variables
    for idx in range(len(var_list)):
        get_deri(deri_list[idx][0], var_list[idx], 0, "+")
        get_deri(deri_list[idx][1], var_list[idx], 0, "-")
    
    #create form for the auxi and source terms
    invg_forms = get_invg_forms(var_list)
    lapse_form = 1./pow(-invg_list[0], 0.5)
    shift_form = -invg_list[1]/invg_list[2] 

    projcet_functions(invg_forms, invg_list) 
    lapse = project(lapse_form, func_space)
    shift = project(shift_form, func_space)
       
    #create form for the Hhat 
    Hhat_forms = get_Hhat_forms(var_list, deri_list, auxi_list, Hhat_list)

    #Runge Kutta step
    
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
