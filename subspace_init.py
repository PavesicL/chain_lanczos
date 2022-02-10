#!/usr/bin/env python3

import numpy as np
import bitwise_ops as bitwise
import utility 
from utility import NchooseK
#import parameters_and_parsing as pp

class SUBSPACE:
	def __init__(self, n, Sz):
		self.n = n
		self.Sz = Sz

		self.nUP = int((n/2) - Sz)
		self.nDOWN = int(n - self.nUP)

		if self.nUP != (n/2) - Sz:
			raise ValueError(f"nUP is not an integer! nUP = {self.nUP}")


class BASIS(SUBSPACE):

	def __init__(self, N, n, Sz):
		SUBSPACE.__init__(self, n, Sz)

		self.N = N

	def basisList(self):
		resList = np.empty(shape=self.size(), dtype=np.int32)
		count=0
		for m in range(2**(2*self.N)):
			if bitwise.countSetBits(m) == self.nUP + self.nDOWN:	#correct number of particles
				if bitwise.spinUpBits(m, self.N) == self.nUP and bitwise.spinDownBits(m, self.N) == self.nDOWN:	#correct number of particles UP and DOWN
					resList[count] = m
					count+=1
		return resList				

	def size(self):
		return int(NchooseK(self.N, self.nUP) * NchooseK(self.N, self.nDOWN))

	def printBasis(self):
		for m in self.basisList():
			binm = utility.binaryString(m, self.N)
			print(f"{m}	{binm}")

###################################################################################################

def make_nrange_list(p):
	#set the n sectors
	if p.refisn0:
		nrange=[int(p.n0 + 0.5-(p.epsimp/p.U))]
	else: 
		nrange=[p.nref]
	i=1
	while i <= p.nrange:
		nrange.append(nrange[0]+i)
		nrange.append(nrange[0]-i)	
		i+=1
	return nrange	


def subspace_dict(p):

	subspaces = {}

	nlist = make_nrange_list(p)

	if p.subspace_set == "basic":
		#only compute Sz = 0 and Sz = 1/2
		for n in nlist:
			Sz = 0.5 * (n%2)
			subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)
	
	elif p.subspace_set == "extended":
		#includes Sz = 1 and Sz = 3/2
		if n%2 == 0:
			for Sz in [0, 1]:
				subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)
		elif n%2 ==1:
			for Sz in [1/2, 3/2]:
				subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)

	elif p.subspace_set == "extended_magnetic":
		#includes Sz = -1, 0, 1 and Sz = -3/2, -1/2, 1/2, 3/2.
		for n in nlist:
			if n%2 == 0:
				for Sz in [-1, 0, 1]:
					subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)
			elif n%2 == 1:
				for Sz in [-3/2, -1/2, 1/2, 3/2]:
					subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)

	elif p.subspace_set == "all":
		#add all Sz subspaces, but assume +/- Sz degeneracy
		for n in nlist:
			if n%2 == 0:
				for Sz in [n/2 -i for i in range((n//2)+1)]:
					subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)

			elif n%2 == 1:
				for Sz in [n/2 -i for i in range((n+1)//2)]:
					subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)
	
	elif p.subspace_set == "all_magnetic":
		#add all subspaces including the -Sz ones
		for Sz in [n/2 -i for i in range(n+1)]:
			subspaces[(n, Sz)] = BASIS(N=p.N, n=n, Sz=Sz)
	
	return subspaces