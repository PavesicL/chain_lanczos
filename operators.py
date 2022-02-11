#!/usr/bin/env python3

import numpy as np
from numba import jit
from bitwise_ops import *
from utility import binaryString
# ELECTRON OPERATORS ##############################################################################

@jit
def prefactor_offset(m, off, N):
	"""
	Calculates the fermionic prefactor for a fermionic operator, acting on site given with the offset off. Sets all succeeding bits to zero and count the rest. 
	"""

	count=0
	for i in range(off+1, 2*N):	#count bits from offset to the end
		count+=bit(m, i)
	
	"""
	#THIS WORKS MUCH SLOWER BUT IS CLEARER TO UNDERSTAND
	#turns off the impurity bit, as it does not contribute
	turnOffImp = (2**(2*N))-1	#this is the number 100000..., where 1 is at the position of the impurity.
	m = m & turnOffImp
	#set bits to zero
	
	#m = clearBitsAfter(m, off, 2*N)

	#count the 1s of the cleared bit
	count = countSetBits(clearBitsAfter(m, off, 2*N))
	"""

	return -(2 * (count%2)-1)


@jit					
def nOP(i, s, m, N):
	"""
	Calculates the application of the counting operator to a basis state m. Returns 0 or 1, according to the occupation of the energy level.
	"""
	return bit(m, 2*(N-i)-1-s)

@jit
def nOpOnState(i, s, state, N, basisList):
	"""	
	Calculates the application of the counting operator on a given state vector.
	"""
	lengthOfBasis = len(basisList)

	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += nOP(i, s, basisList[k], N)*coef
	
	return new_state		

@jit
def nTotOnState(i, state, N, basisList):
	"""	
	Calculates the application of the counting operator on a given state vector without spin.
	"""
	lengthOfBasis = len(basisList)
	
	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += nOP(i, 0, basisList[k], N)*coef
			new_state[k] += nOP(i, 1, basisList[k], N)*coef
	
	return new_state

@jit
def crcranan(i, j, m, N):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a basis vector m, given as an integer.
	"""

	offi = 2*(N-i)-1	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-j)-1

	#at each step, the if statement gets TRUE if the operation is valid (does not destroy the state)
	m1 = flipBit(m, offj-0)
	if m>m1:
		m2 = flipBit(m1, offj-1)
		if m1>m2:
			m3 = flipBit(m2, offi-1)
			if m2<m3:
				m4 = flipBit(m3, offi)
				if m3<m4:
					return m4
	return 0  

@jit
def crcrananOnState(i, j, state, N, basisList):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a state.
	"""
	lengthOfBasis = len(basisList)

	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):

		coef = state[k]
		if coef!=0:

			m = crcranan(i, j, basisList[k], N)

			if m!=0:
				"""
				THIS IS ONE OF THE BOTTLENECKS - given a resulting state (m), find which index in basisList it corresponds to. 
				The solution with the dictionary (of type { state : index_in_basisList }) turns out to be slow. So is the usage 
				of the numpy function np.where(), which finds all occurences of a given value in a list. The current solution is 
				using searchsorted, a function which returns (roughly) the first position of the given value, but needs the list 
				to be sorted. basisList is sorted by construction, so this works. 
				"""
				res = np.searchsorted(basisList, m)

				new_state[res] += coef
				
	return new_state
	
# HOPPING OPERATORS ###############################################################################

@jit
def hoppingOp(i, j, s, m, N):
	"""
	This is c^dag_i,s c_j,s
	"""

	offi = 2*(N-i)-1-s	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-j)-1-s
	
	if offi<0 or offj<0:
		raise ValueError("OFFSET SMALLER THAN ZERO!")

	if bit(m, offi)==0:
		if bit(m, offj)==1:
			m = flipBit(m, offj)
			prefactor = prefactor_offset(m, offj, N)
			m = flipBit(m, offi)
			prefactor *= prefactor_offset(m, offi, N)

			return prefactor, m
	return 0, 0

@jit
def hoppingOnState(i, j, state, N, basisList):
	"""
	Calculates the application of c^dag_i c_j + c^dag_j c_i on a given state vector.
	"""
	lengthOfBasis = len(basisList)

	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			for s in [0, 1]:
				prefac1, m1 = hoppingOp(i, j, s, basisList[k], N)
				prefac2, m2 = hoppingOp(j, i, s, basisList[k], N)

				if m1!=0:
					new_state[np.searchsorted(basisList, m1)] += coef * prefac1 
				if m2!=0:
					new_state[np.searchsorted(basisList, m2)] += coef * prefac2

	return new_state		

# SPIN OPERATORS ##################################################################################

def SpSmOp(i, j, m, N):

	offiup = 2*(N-i)-1-1
	offidn = 2*(N-i)-1-0
	offjup = 2*(N-j)-1-1
	offjdn = 2*(N-j)-1-0

	#first Sm:
	if bit(m, offjup)==1:
		if bit(m, offjdn)==0:
			m = flipBit(m, offjup)
			prefactor = prefactor_offset(m, offjup, N)
			m = flipBit(m, offjdn)
			prefactor *= prefactor_offset(m, offjdn, N)
		else:
			return 0, 0
	else:
		return 0, 0

	#then Sp:
	if bit(m, offidn)==1:
		if bit(m, offiup)==0:
			m = flipBit(m, offidn)
			prefactor *= prefactor_offset(m, offidn, N)
			m = flipBit(m, offiup)
			prefactor *= prefactor_offset(m, offiup, N)
		else:
			return 0, 0
	else:
		return 0, 0
	return prefactor, m	

def SmSpOp(i, j, m, N):

	offiup = 2*(N-i)-1-1
	offidn = 2*(N-i)-1-0
	offjup = 2*(N-j)-1-1
	offjdn = 2*(N-j)-1-0

	#first Sp:
	if bit(m, offjdn)==1:
		if bit(m, offjup)==0:
			m = flipBit(m, offjdn)
			prefactor = prefactor_offset(m, offjdn, N)		
			m = flipBit(m, offjup)
			prefactor *= prefactor_offset(m, offjup, N)
		else:
			return 0, 0
	else:
		return 0, 0	

	#then Sm:
	if bit(m, offiup)==1:
		if bit(m, offidn)==0:
			m = flipBit(m, offiup)
			prefactor *= prefactor_offset(m, offiup, N)
			m = flipBit(m, offidn)
			prefactor *= prefactor_offset(m, offidn, N)			
		else:
			return 0, 0
	else:
		return 0, 0
	return prefactor, m	
	
def SpSmOnState(i, j, state, N, basisList):
	"""
	Calculates the application of S^+ S^- on a given state vector.
	"""
	lengthOfBasis = len(basisList)
	new_state = np.zeros(lengthOfBasis)
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			prefac, m = SpSmOp(i, j, basisList[k], N)
			if m!=0:
				new_state[np.searchsorted(basisList, m)] += coef * prefac
	return new_state

def SmSpOnState(i, j, state, N, basisList):
	"""
	Calculates the application of S^+ S^- on a given state vector.
	"""
	lengthOfBasis = len(basisList)
	new_state = np.zeros(lengthOfBasis)
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			prefac, m = SmSpOp(i, j, basisList[k], N)
			if m!=0:
				new_state[np.searchsorted(basisList, m)] += coef * prefac
	return new_state

def SScorrelationOnState(i, j, state, params, basisList):
	"""
	Spin correlations <S_i S_j>
	"""

	nupnup = nOpOnState(i, 0, nOpOnState(j, 0, state, params.N, basisList), params.N, basisList)
	nupndn = nOpOnState(i, 0, nOpOnState(j, 1, state, params.N, basisList), params.N, basisList)
	ndnnup = nOpOnState(i, 1, nOpOnState(j, 0, state, params.N, basisList), params.N, basisList)
	ndnndn = nOpOnState(i, 1, nOpOnState(j, 1, state, params.N, basisList), params.N, basisList)
	
	SzSz = 0.25 * ( nupnup - nupndn - ndnnup + ndnndn)

	SpSm = SpSmOnState(i, j, state, params.N, basisList)
	SmSp = SmSpOnState(i, j, state, params.N, basisList)

	return SzSz + 0.5 * SpSm + 0.5 * SmSp
