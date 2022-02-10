#!/usr/bin/env python3

import numpy as np
from scipy.sparse.linalg import LinearOperator
import operators_constructed as op 

###################################################################################################

class QD:
	"""
	Class containing impurity parameters.
	"""

	def __init__(self, level, U, epsimp, EZ):
		self.level = level
		self.U = U
		self.epsimp = epsimp
		self.nu = 0.5 - (epsimp/U)
		self.EZ = EZ

class SC:
	"""
	Class containing SC parameters.
	"""

	def __init__(self, levels, Nbath, D, alpha, Ec, n0, EZ):
		self.levels = levels #levels in the whole system that this island occupies 
		self.D = D
		self.Nbath = Nbath
		self.d = 2*D/Nbath
		self.alpha = alpha
		self.Ec = Ec
		self.n0 = n0
		self.EZ = EZ

	def eps(self, i, band_level_shift=True, flat_band=False):
		# ONE BASED!!!

		ishift = self.levels[0]-1 
		truei = i - ishift #so that if the SC island is on levels [3, 4, 5], eps(i=3) gives the energy for the lowest level, ie. eps(1) = -1/2.

		if band_level_shift:
				shift = - self.alpha * self.d/2
		else:
			shift = 0
		if flat_band:
			if i >= self.Nbath/2:
				return -0.5 + shift
			else:
				return +0.5 + shift	
		else:	
			return -self.D + (truei-0.5)*self.d + shift

class HYB:
	"""
	Class containing hybridization parameters between a QD and a SC.
	"""
	def __init__(self, Gamma, V):
		self.Gamma = Gamma
		self.V = V

	def v(self, Nbath):	
		return np.sqrt( 2 * self.Gamma / (np.pi * Nbath) )

class HOP:
	"""
	Class containing hopping parameters between two QDs.
	"""
	def __init__(self, t, V):
		self.t = t
		self.V = V
		#anything else?

# LIN OP SETUP ####################################################################################

class HAMILTONIAN(LinearOperator):
	"""
	This is a class, built-in to scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	The operator can be diagonalised using the function scipy.sparse.linalg.eigsh().
	"""	
	def __init__(self, parameters, basis):
		self.shape = (basis.size(), basis.size())
		self.dtype = np.dtype("float64")

		self.b = basis
		self.p = parameters

	def _matvec(self, state):
		return self.HonState(state, self.p, self.b)				

	def HonState(self, state, p, b):	
		"""
		INPUT:
		state 	- a vector representing the wavefunction
		p 		- instance of class PARAMETERS, holding all model parameters
		b 		- instance of class BASIS, holding the basis and subspace parameters
		"""
		
		impurity, SCisland, interaction = np.zeros(b.size()), np.zeros(b.size()), np.zeros(b.size())

		countParts = 0
		for part in p.chain:
			countParts += 1 # countParts is 1-based!!!
			
			if isinstance(part, QD):
				impurity += op.impurityEnergyOnState(state, part, p, b)

			elif isinstance(part, SC):
				SCisland += op.SCIslandEnergyOnState(state, part, p, b)

			if countParts < len(p.chain):
				prevPart = p.chain[countParts-1]
				thisPart = p.chain[countParts]
				coupling = p.couplings[countParts-1]

				interaction += op.interactionOnState(state, prevPart, thisPart, coupling, p, b)	

		return impurity + SCisland + interaction