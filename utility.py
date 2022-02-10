#!/usr/bin/env python3

import bitwise_ops as bitwise
from math import factorial

def printVector(p, vector, basisList):
	"""
	Prints the most prominent elements of a vector and their amplitudes.
	"""

	for i in range(len(vector)):
		if abs(vector[i])>p.vector_precision:
			#print(basisList[i], bin(basisList[i]), vector[i])
			sketch = sketchBasisVector(p.N, basisList[i])

			print(basisList[i], format(basisList[i], "0{}b".format(2*p.N)), sketch, vector[i])

def binaryString(m, N):
	return format(m, "0{}b".format(2*N))

def checkParams(N, n, nwimpUP, nwimpDOWN):
	"""
	Checks if the parameter values make sense.
	"""
	allOK = 1

	if n>2*N:
		print("WARNING: {0} particles is too much for {2} levels!".format(n, N))
		allOK=0

	if nwimpUP + nwimpDOWN != n:
		print("WARNING: some mismatch in the numbers of particles!")
		allOK=0

	if allOK:
		print("ns check OK.")

def setAlpha(N, d, dDelta):
	"""
	Returns alpha for a given dDelta. 
	"""
	omegaD = 0.5*N*d	
	Delta = d/dDelta
	return 1/(np.arcsinh(omegaD/(Delta)))

def saveToFile(savelist, fname):

	with open(fname, "w") as ff:
		for ll in savelist:
			ff.write("	".join([str(i) for i in ll]) + "\n") # works with any number of elements in a line
		
def checkLevel(basisVector, i, N):
	"""
	Checks the occupancy of the i-th energy level.
	"""
	off = 2*(N-i)-1	#offset without spin

	up, down = False, False
	if bitwise.bit(basisVector, off-0):
		up=True
	if bitwise.bit(basisVector, off-1):
		down=True

	if up and down:
		return "2"
	elif up:
		return "UP"
	elif down:
		return "DO"
	else:
		return "0"	

def NchooseK(n, k):
	return factorial(n) / (factorial(k) * factorial(n-k))

def nSzi_string(n, Sz, i):
	return f"{n}/{Sz}/{i}/"
