from __future__ import print_function
from dolfin import *
import matplotlib.pyplot as plt
from hdw import *
import sch_kerr_schild as sks
import rk2
import sys
import getopt

def main():
    """
    main computating process
    """
    DG_degree = 2
    mesh_num = 10
    plot_idx = 0

    opts, dumps = getopt.getopt(sys.argv[1:], "-m:-d:-i:")
    for opt, arg in opts:
        if opt == "-m":
            mesh_num = int(arg)
        if opt == "-d":
            DG_degree = int(arg)
        if opt == "-i":
            plot_idx = int(arg)

    #create mesh and define function space
    mesh_len = 2.0
    inner_bdry = 1.0
    mesh = IntervalMesh(mesh_num, inner_bdry, inner_bdry + mesh_len)
    func_space = FunctionSpace(mesh, "DG", DG_degree)
    dt = 0.3*(mesh_len/mesh_num)**DG_degree

    #coordinate function
    r = SpatialCoordinate(mesh)[0]

    #define functions for the variables
    var_list = sks.get_var_list(func_space)
    deri_list = [[Function(func_space), Function(func_space)] for dummy in range(len(var_list))]
    H_list = sks.get_H_list(func_space)
    deriH_list = sks.get_deriH_list(func_space)
    #define functions for the auxi variables
    boundary_list = [Function(func_space) for dummy in range(15)]
    project_functions(var_list, boundary_list)
    
    invg_list = [Function(func_space) for dummy in range(3)]
    auxi_list = [Function(func_space) for dummy in range(5)]
    gamma_list = [Function(func_space) for dummy in range(8)]
    T_list = [Function(func_space) for dummy in range(4)]
    C_list = [Function(func_space) for dummy in range(2)]

    Hhat_list = [Function(func_space) for dummy in range(15)]
    src_list = [Function(func_space) for dummy in range(15)]
    rhs_list = [Function(func_space) for dummy in range(15)]

    #create form for middle terms
    invg_forms = get_invg_forms(var_list)
    auxi_forms = get_auxi_forms(var_list, invg_list) 
    gamma_forms = get_gamma_forms(var_list, invg_list, auxi_list, r)
    C_forms = get_C_forms(H_list, gamma_list)
    T_forms = get_T_forms(var_list, invg_list, auxi_list)

    Hhat_forms = get_Hhat_forms(var_list, deri_list, auxi_list)
    src_forms = get_source_forms(var_list, invg_list, gamma_list, auxi_list, T_list, C_list, H_list, deriH_list, r)
    rhs_forms = get_rhs_forms(Hhat_forms, src_forms)
    
    #pack forms and functions
    form_packs = (invg_forms, auxi_forms, gamma_forms, C_forms, T_forms, Hhat_forms, src_forms, rhs_forms) 
    func_packs = (invg_list, auxi_list, gamma_list, C_list, T_list, Hhat_list, src_list, rhs_list)
    
    #Runge Kutta step
    temp_var_list = [Function(func_space) for dummy in range(15)]
    exact_var_list = [Function(func_space) for dummy in range(15)]
    project_functions(var_list, exact_var_list)
    t_now = 0.0
    t_end = 100.0
    time_seq = []
    error_rhs_seq = [[] for dummy in range(15)]
    error_var_seq = [[] for dummy in range(15)]
    error_C0_seq = []
    error_C1_seq = []
    zero_func = project(Expression("0.", degree=10), func_space)

    dif_forms = [var_list[idx] - exact_var_list[idx] for idx in range(15)]
    dif_list = [Function(func_space) for dummy in range(15)]
    plt.ion()

    while t_now + dt <= t_end:
        project_functions(var_list, temp_var_list)
        rk2.rk2(var_list, boundary_list, temp_var_list, deri_list, form_packs, func_packs, dt, t_now)

        error_rhs = [errornorm(rhs, zero_func, 'L2') for rhs in rhs_list]
        error_C0 = errornorm(C_list[0], zero_func, 'L2')
        error_C1 = errornorm(C_list[1], zero_func, 'L2')
        error_var = [errornorm(var_list[idx], exact_var_list[idx], 'L2') for idx in range(len(var_list))]
        error_C0_seq.append(error_C0)
        error_C1_seq.append(error_C1)
        for idx in range(15):
            error_rhs_seq[idx].append(error_rhs[idx])
            error_var_seq[idx].append(error_var[idx])

        t_now += dt
        time_seq.append(t_now)

        plt.clf()
        
        project_functions(dif_forms, dif_list)
        plt.subplot(3, 3, 1)
        plot(dif_list[plot_idx]) 
        plt.title('error of var_'+str(plot_idx))
        
        plt.subplot(3, 3, 2)
        plot(dif_list[plot_idx+3])
        plt.title('error of var_'+str(plot_idx+3))

        plt.subplot(3, 3, 3)
        plot(dif_list[plot_idx+6])
        plt.title('error of var_'+str(plot_idx+6))

        plt.subplot(3, 3, 4)
        plt.plot(time_seq, error_var_seq[plot_idx], 'r')
        plt.title('L2 error of var_'+str(plot_idx)+' evolved in time')

        plt.subplot(3, 3, 5)
        plt.plot(time_seq, error_var_seq[plot_idx+3], 'r')
        plt.title('L2 error of var_'+str(plot_idx+3)+' evolved in time')

        plt.subplot(3, 3, 6)
        plt.plot(time_seq, error_var_seq[plot_idx+6], 'r')
        plt.title('L2 error of var_'+str(plot_idx+6)+' evolved in time')

        plt.subplot(3, 3, 7)
        plot(C_list[0])
        plt.title('C0')

        plt.subplot(3, 3, 8)
        plot(C_list[1])
        plt.title('C1')

        plt.subplot(3, 3, 9)
        plt.plot(time_seq, error_C0_seq, 'r')
        plt.title('L2 norm of constraint C0')

        plt.pause(0.1)


    plt.pause(100000000)
    if t_now < t_end:
        project_functions(var_list, temp_var_list)
        rk2.rk2(var_list, boundary_list, temp_var_list, deri_list, form_packs, func_packs, t_end-t_now, t_now)

        error_rhs = [errornorm(rhs, zero_func, 'L2') for rhs in rhs_list]
        error_C0 = errornorm(C_list[0], zero_func, 'L2')
        error_C1 = errornorm(C_list[1], zero_func, 'L2')
        error_var = [errornorm(var_list[idx], exact_var_list[idx], 'L2') for idx in range(len(var_list))]
        error_C0_seq.append(error_C0)
        error_C1_seq.append(error_C1)
        for idx in range(15):
            error_rhs_seq[idx].append(error_rhs[idx])
            error_var_seq[idx].append(error_var[idx])

        t_now += dt
        time_seq.append(t_now)

        plt.clf()
        
        plt.subplot(3, 3, 1)
        plot(var_list[plot_idx]) 
        plt.title('var_'+str(plot_idx))
        
        plt.subplot(3, 3, 2)
        plot(var_list[plot_idx+3])
        plt.title('var_'+str(plot_idx+3))

        plt.subplot(3, 3, 3)
        plot(var_list[plot_idx+6])
        plt.title('var_'+str(plot_idx+6))

        plt.subplot(3, 3, 4)
        plt.plot(time_seq, error_var_seq[plot_idx], 'r')
        plt.title('L2 error of var_'+str(plot_idx))

        plt.subplot(3, 3, 5)
        plt.plot(time_seq, error_var_seq[plot_idx+3], 'r')
        plt.title('L2 error of var_'+str(plot_idx+3))

        plt.subplot(3, 3, 6)
        plt.plot(time_seq, error_var_seq[plot_idx+6], 'r')
        plt.title('L2 error of var_'+str(plot_idx+6))

        project_functions(dif_forms, dif_list)
        plt.subplot(3, 3, 7)
        plot(dif_list[plot_idx])
        plt.title( label='error of var_'+str(plot_idx))

        plt.subplot(3, 3, 8)
        plot(C_list[0])
        plt.title('C0')

        plt.subplot(3, 3, 9)
        plt.plot(time_seq, error_C0_seq, 'r')
        plt.title('L2 norm of constraints C0')

    plt.pause(100000)
    plt.ioff()
    #test_list = [Function(func_space) for dummy in range(8)]
    #test_forms = get_test_forms(var_list, invg_list, gamma_list, auxi_list, T_list, C_list, H_list, deriH_list, r)
    #project_functions(test_forms, test_list)

    #for idx in range(len(var_list)):
    #    get_deri(deri_list[idx][0], var_list[idx], boundary_list[idx], 0, "+")                        
    #    get_deri(deri_list[idx][1], var_list[idx], boundary_list[idx], 0, "-") 

    #for idx in range(len(form_packs)):
    #    project_functions(form_packs[idx], func_packs[idx]) 
     
    for rhs_idx in range(len(rhs_list)):
        print(" ")
        for vert in vertices(mesh):
            idx = vert.index() 
            print("rhs[" + str(rhs_idx) + "]" + str(mesh.coordinates()[idx])
                     + "=" + str(rhs_list[rhs_idx].compute_vertex_values()[idx]))
            if idx >= 2:
                break


    #for test_idx in range(len(test_list)):
    #    print(" ")
    #    for vert in vertices(mesh):
    #        idx = vert.index() 
    #        print("test[" + str(test_idx) + "]" + str(mesh.coordinates()[idx])
    #                 + "=" + str(test_list[test_idx].compute_vertex_values()[idx]))
    #        if idx >=1:
    #            break

    #print(time_seq)
    #for idx in range(15):
    #    print(error_rhs_seq[idx])
    #    print(error_var_seq[idx])
    # plt.xlabel('time')
    # plt.ylabel('error of rhs[' + str(0) +']')
    # title = ''
    # plt.title(title)  


    # plot(invg_list[2])
    # plot(invg_list[0])
    # plot(mesh)
    # plt.show()

main()
