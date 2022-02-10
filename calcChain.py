#!/usr/bin/env python3

import sys
import time
import h5py
from scipy.sparse.linalg import eigsh

import parameters_and_parsing as pp
import subspace_init
import physical_classes as physics
import utility
import result_printing as res
###########################################################
if len(sys.argv)!=2:
	print("Usage: {0} inputFile".format(sys.argv[0]))
	exit()

inputFile = sys.argv[1]

###########################################################
#print the input data

p = pp.PARAMETERS(inputFile)		#all problem parameters are stored here
subspaceDict = subspace_init.subspace_dict(p)	#dictionary of all subspaces, contains ther bases

print()
for key in vars(p).keys():
	if key != "chain" and key != "couplings" and key != "inputList":
		print(f"{key} = {vars(p)[key]}")

###########################################################
start = time.time() #start timer


results = {}
for subspace in subspaceDict:
	s = subspaceDict[subspace]
	
	print()
	print(f"Diagonalizing: {s.n}, {s.Sz}")	
	print("basis lenght:", s.size())
	
	H = physics.HAMILTONIAN(p, s)

	values, vectors = eigsh(H, k=min(s.size()-1, p.get_num_of_states(s)), which="SA")
	vectors = vectors.T
	values = values + p.Eshift	

	for i in range(len(values)):
		results[(s.n, s.Sz, i)] = res.STATE(s.n, s.Sz, i, values[i], vectors[i], s)		


h5file = h5py.File("solution.h5", "w")
res.print_and_save_results(results, p, h5file)

###########################################################

end = time.time()
print(f"\nWall time: {round(end - start, 1)} s")
