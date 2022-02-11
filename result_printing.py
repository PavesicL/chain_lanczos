#!/usr/bin/env python3

from subspace_init import BASIS
import compute_properties as cp
import utility as util
import numpy as np
import physical_classes as physics
from scipy.sparse.linalg import eigsh

class STATE(BASIS):

	def __init__(self, n, Sz, i, E, psi, basis):
		BASIS.__init__(self, basis.N, n, Sz)
		self.n = n
		self.Sz = Sz
		self.i = i

		self.E = E
		self.psi = psi

def diagonalize(s, p):

	results = {}

	print()
	print(f"Diagonalizing: {s.n}, {s.Sz}")	
	print("basis lenght:", s.size())
	
	H = physics.HAMILTONIAN(p, s)

	values, vectors = eigsh(H, k=min(s.size()-1, p.get_num_of_states(s)), which="SA")
	vectors = vectors.T
	values = values + p.Eshift	

	for i in range(len(values)):
		results[(s.n, s.Sz, i)] = STATE(s.n, s.Sz, i, values[i], vectors[i], s)		

	return results

def print_and_save_results(results, params, h5file):

	print("\n###########################################################################\n")

	for state in sorted(results):
		r = results[state]
	
		print(f"\nSTATE n = {r.n}, Sz = {r.Sz}, i = {r.i}:")
		
		print_and_save_Es(r, params, h5file)
		print_site_occupations(r, params, h5file)
		if params.print_vector:
			print_vector(r, params)
		if params.spin_correlation_matrix:
			print_and_save_spin_correlation_matrix(r, params, h5file)	
		if params.end_to_end_spin_correlations:
			print_and_save_end_to_end_spin_correlations(r, params, h5file)

	print("\n###########################################################################")
			

###################################################################################################

def print_list(ll):
	print(*ll, sep=", ")

###################################################################################################

def print_and_save_Es(state, params, h5file):
	print(f"E = {state.E}")
	h5save(h5file, state, "E", state.E)

# VECTOR PRINT ####################################################################################

def sketchBasisVector(N, basisVector):
	"""
	Given a basis vector, sketch its level occupanies.
	"""
	Vstring = ""
	for i in range(N):
		Vstring += utility.checkLevel(basisVector, i, N) + "|"

	return Vstring	

def print_vector(state, params):
	"""
	Prints the most prominent elements of a vector and their amplitudes.
	"""
	psi = state.psi
	for i in range(len(psi)):
		if abs(psi[i])>params.print_vector_precision:
			sketch = sketchBasisVector(params.N, state.basisList()[i])
			print(state.basisList()[i], format(state.basisList()[i], "0{}b".format(2*params.N)), sketch, psi[i])

# OCCUPATIONS #####################################################################################

def print_site_occupations(state, params, h5file):
	occs = cp.compute_site_occupations(state.psi, state.basisList(), params)
	print("OCCUPATIONS:")
	print_list(occs)
	h5save(h5file, state, "occupation", occs)


def print_and_save_end_to_end_spin_correlations(state, params, h5file):
	scorr = cp.compute_end_to_end_spin_correlation(state.psi, state.basisList(), params)
	print(f"END-TO-END SPIN CORRELATIONS: {scorr}")
	h5save(h5file, state, "S1SN_correlation", scorr)


def print_and_save_spin_correlation_matrix(state, params, h5file):
	smat = cp.compute_spin_correlation_matrix(state.psi, state.basisList(), params)
	matsum = np.sum(smat)
	print(f"SPIN-SPIN CORRELATION MATRIX COMPUTED. SUM = {matsum}")
	h5save(h5file, state, "spin_correlation_matrix", smat)


# SAVE TO HDF5 FILE ###############################################################################

def h5save(file, state, saveString, values):
	file.create_dataset(util.nSzi_string(state.n, state.Sz, state.i) + saveString + "/", data=values)
