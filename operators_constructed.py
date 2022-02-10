#!/usr/bin/env python3

import numpy as np
from numba import jit, int32, float64
import physical_classes
from operators import *

# IMPURITY ########################################################################################
#@jit
def impurityEnergyOnState(state, qd, p, b):
	"""
	Calculates the contribution of the impurity to energy (kinetic and potential energy).
	INPUT:
	state 	- vector representing the wavefunction to act on
	qd 		- instance of class QD holding its parameters
	p 		- instance of class PARAMETERS
	b 		- instance of class BASIS
	OUTPUT:
	new_state - the resulting wavefunction
	"""

	basisList = b.basisList()

	nimpUP = nOpOnState(qd.level, 0, state, p.N, basisList)
	nimpDOWN = nOpOnState(qd.level, 1, state, p.N, basisList)
	nimpUPnimpDOWN = nOpOnState(qd.level, 0, nOpOnState(qd.level, 1, state, p.N, basisList), p.N, basisList)
	
	new_state = qd.epsimp*(nimpUP + nimpDOWN) + qd.U*nimpUPnimpDOWN + 0.5*qd.EZ*(nimpUP - nimpDOWN)
	return new_state

# SC ISLAND #######################################################################################

#@jit()
def SCIslandEnergyOnState(state, sc, p, b):
	"""
	Calculates the contribution of the SC island Hamiltonian. 
	INPUT:
	state 	- vector representing the wavefunction to act on
	sc 		- instance of class SC holding its parameters
	p 		- instance of class PARAMETERS
	b 		- instance of class BASIS
	OUTPUT:
	new_state - the resulting wavefunction
	"""

	basisList = b.basisList()

	interaction, charging, kinetic, magnetic = np.zeros(b.size()), np.zeros(b.size()), np.zeros(b.size()), np.zeros(b.size())
	nSC, nSC2 = np.zeros(b.size()), np.zeros(b.size())
	new_state = np.zeros(b.size())

	for i in sc.levels:
		niUP = nOpOnState(i, 0, state, p.N, basisList)
		niDOWN = nOpOnState(i, 1, state, p.N, basisList)

		kinetic += sc.eps(i, p.band_level_shift, p.flat_band) * (niUP + niDOWN)
		magnetic += 0.5 * sc.EZ * niUP
		magnetic +=  - 0.5 * sc.EZ * niDOWN

		if sc.Ec != 0:
			#nSC += nTotOnState(i, state, p.N, b)
			nSC += niUP + niDOWN			

		for j in sc.levels:
			interaction += crcrananOnState(i, j, state, p.N, basisList)
			if sc.Ec!=0:
				nSC2 += nTotOnState(i, (nTotOnState(j, state, p.N, basisList)), p.N, basisList)

	if sc.Ec!=0:
		charging += nSC2 - 2 * sc.n0 * nSC

	new_state += kinetic + magnetic + (sc.d * sc.alpha * interaction) + sc.Ec * charging
	return new_state

# imp - SC HOPPING ################################################################################

#@jit
def interactionOnState(state, first, second, coupling, p, b):
	"""
	A wrapper for interaction calculation between two parts of the system. 
	If these are both QDs, use hopping, otherwise, use hybridization.
	"""
	
	hopping, capacitive = np.zeros(b.size()), np.zeros(b.size())

	if isinstance(first, physical_classes.QD) and isinstance(second, physical_classes.QD):
		hopping += QDinteractionOnState(state, fist, second, coupling, p, b)
		capacitive += QDcapacitiveOnState(state, first, second, coupling, p, b)

	else:
		if isinstance(first, physical_classes.QD) and isinstance(second, physical_classes.SC):
			hopping += QDSCinteractionOnState(state, first, second, coupling, p, b)
			capacitive += QDSCcapacitiveOnState(state, first, second, coupling, p, b)	
		elif isinstance(first, physical_classes.SC) and isinstance(second, physical_classes.QD):
			hopping += QDSCinteractionOnState(state, second, first, coupling, p, b)
			capacitive += QDSCcapacitiveOnState(state, second, first, coupling, p, b)

	return hopping + capacitive		

# QD-QD interaction ###############################################################################
#TO DO!
#@jit
def QDinteractionOnState(state, qd1, qd2, hop, p, b):
	#identity for now
	return state
#@jit
def QDcapacitiveOnState(state, qd1, qd2, hop, p, b):
	#identity
	return state

# QD-SC interaction ###############################################################################

#@jit
def QDSCinteractionOnState(state, qd, sc, hyb, p, b):
	"""
	Hopping between a qd and sc island.
	"""
	basisList = b.basisList()
	hopping = np.zeros(b.size())
	for i in sc.levels:
		v = hyb.v(sc.Nbath)
		hopping += v * hoppingOnState(qd.level, i, state, p.N, basisList)

	return hopping

#@jit
def QDSCcapacitiveOnState(state, qd, sc, hyb, p, b):
	"""
	Capacitive coupling between qd and sc island.
	V (nsc - n0)(nimp - nu) = V (nsc*nimp - n0*nimp - nu*nimp + n0*nu)
	"""
	basisList = b.basisList()

	quadratic, linearQD, linearSC = np.zeros(b.size()), np.zeros(b.size()), np.zeros(b.size())
	n0nu = qd.nu * sc.n0 * np.ones(b.size())

	linearQD += nTotOnState(qd.level, state, p.N, basisList) # nimp

	for i in sc.levels:
		linearSC += nTotOnState(i, state, p.N, basisList) # nSC
		quadratic += nTotOnState(i, linearQD, p.N, basisList) # nSC * nimp

	return hyb.V * (quadratic - sc.n0 * linearQD - qd.nu * linearSC + n0nu) 

