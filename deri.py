from fenics import *

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

