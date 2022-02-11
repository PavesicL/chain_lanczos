#!/usr/bin/env python3

import h5py
from subspace_init import BASIS
from operators import *

def compute_site_occupations(state, basisList, params):

	occupations = []

	lengthOfBasis = len(basisList)

	for i in range(params.N):
		npsi = nOpOnState(i, 0, state, params.N, basisList) + nOpOnState(i, 1, state, params.N, basisList)
		occ = np.dot(state, npsi)

		occupations.append(occ)
	return occupations

def compute_end_to_end_spin_correlation(state, basisList, params):
	SSpsi = SScorrelationOnState(0, params.N-1, state, params, basisList)
	SS = np.dot(state, SSpsi)
	return SS

def compute_spin_correlation_matrix(state, basisList, params):
	mat = np.zeros((params.N, params.N))
	for i in range(params.N):
		for j in range(params.N):
			SSpsi = SScorrelationOnState(i, j, state, params, basisList)
			SS = np.dot(state, SSpsi)
			mat[i, j] = SS
	return mat


