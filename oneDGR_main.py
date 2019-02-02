from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
from deri import *
import sys
import getopt

def main():
    """
    main computational process
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
    g_00 = Function(func_space) 
    g_01 = Function(func_space)
    g_11 = Function(func_space)
    Pi_00 = Function(func_space) 
    Pi_01 = Function(func_space)
    Pi_11 = Function(func_space)
    Phi_00 = Function(func_space) 
    Phi_01 = Function(func_space)
    Phi_11 = Function(func_space)
    S = Function(func_space)
    Pi_S = Function(func_space)
    Phi_S = Function(func_space)
    psi = Function(func_space)
    Pi_psi = Function(func_space)
    Phi_psi = Function(func_space)

    var_list = [g_00, g_01, g_11,
                Pi_00, Pi_01, Pi_11,
                Phi_00, Phi_01, Phi_11,
                S, Pi_S, Phi_S,
                psi, Pi_psi, Phi_psi]
    
    #initialize variables
    u_n = Expression("exp(x[0]) + 5.0", degree=3)
    deri_list = []
    for idx in range(len(var_list)):
        var_list[idx] = project(u_n, func_space)
        deri_list.append([Function(func_space), Function(func_space)])
    
    #compute the derivative of variables
    for idx in range(len(var_list)):
        get_deri(deri_list[idx][0], var_list[idx], 0, "+")
        get_deri(deri_list[idx][1], var_list[idx], 0, "-")

    
    #define variational problem
    
    p = deri_list[5][1] 
    for vert in vertices(mesh):
        idx = vert.index() 
        print(str(mesh.coordinates()[idx]) + "=" + str(p.compute_vertex_values()[idx]))
    
    #error_L2 = errornorm(u_x, p, 'L2')
    #print("L2 error of p_+  = ", error_L2)
    
    plot(p)
    plot(mesh)
    plt.yscale("log")
    plt.show()

main()
