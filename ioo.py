from dolfin import *

def read_var_from_files(var_list):
    for idx in range(len(var_list)):
        var = var_list[idx]
        ufile = HDF5File(MPI.comm_world, var.name()+'.hdf5', 'r')
        ufile.read(var, var.name()) 
        ufile.close()

def write_seqs_to_file(fn, seqs):
    with open(fn, 'w') as f:
        for idx_time in range(len(seqs[0])):
            li = []
            for idx_obj in range(len(seqs)):
                li.append(str(seqs[idx_obj][idx_time])) 
            f.write(' '.join(li)+'\n')

def read_seqs_from_file(fn, seqs):
    with open(fn, 'r') as f:
        lines = f.readlines()
    for line in lines:
        values = line.split(" ")
        for idx in range(len(seqs)):
            seqs[idx].append(float(values[idx]))
