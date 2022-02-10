#!/usr/bin/env python3

import sys
import time
import h5py
from joblib import Parallel, delayed

import parameters_and_parsing as pp
import subspace_init
import utility as util
from result_printing import *
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


if p.parallel:
	num_cores=len(subspaceDict)
	res = Parallel(n_jobs=num_cores)(delayed(diagonalize)(subspaceDict[subspace], p) for subspace in subspaceDict)
else:
	res = [diagonalize(subspaceDict[subspace], p) for subspace in subspaceDict]

# Diagonalize(s, p) returns a dictionary of STATE objects, which are calculated states. res is a list of these for all subspaces.
# Now combine all dictionaries into one big one.

results = util.merge_dicts(*res)

h5file = h5py.File("solution.h5", "w")
print_and_save_results(results, p, h5file)

###########################################################

end = time.time()
print(f"\nWall time: {round(end - start, 1)} s")
