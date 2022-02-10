#!/usr/bin/env python3
from numba import jit

@jit
def flipBit(n, offset):
	"""Flips the bit at position offset in the integer n."""
	mask = 1 << offset
	return(n ^ mask)

@jit
def countSetBits(m): 
	"""Counts the number of bits that are set to 1 in a given integer."""
	count = 0
	while (m): 
		count += m & 1
		m >>= 1
	return count 

@jit
def bit(m, off):
	"""
	Returns the value of a bit at offset off in integer m.
	"""
	if m & (1 << off):
		return 1
	else:
		return 0

@jit
def spinUpBits(m, N):
	"""
	Counts the number of spin up electrons in the state.
	"""
	count=0
	for i in range(1, 2*N, 2):
		if bit(m, i)==1:
			count+=1
	return count		

@jit
def spinDownBits(m, N, allBits=False):
	"""
	Counts the number of spin down electrons in the state.
	"""
	count=0
	for i in range(0, 2*N, 2):
		if bit(m, i)==1:
			count+=1	
	return count	

@jit
def clearBitsAfter(m, off, length):
	"""Clears all bits of a number m with length length with offset smaller OR EQUAL off. Used to determine the fermionic +/- prefactor."""
	clearNUm = 0
	for i in range(off+1, length):
		clearNUm += 2**(i)

	return m & clearNUm
