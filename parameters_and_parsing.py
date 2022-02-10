#!/usr/bin/env python3

################################################################################

import re

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

import numba
from numba import jit, njit
from numba import types
from numba import int32, float64, float32, int64, boolean, char

from physical_classes import *

# PARSING FUNCTIONS ###############################################################################

def getInputList(file):
	"""
	Reads the file into a list, with newline characters removed.
	"""
	with open(file, "r") as f:
		linesList = []
		for line in f:
			if len(line.strip()) > 0:
				linesList.append(line.strip())	
	return linesList

def convertType(val, whichtype):
	if whichtype == "string":
		return val
	elif whichtype == "int":
		return int(val)
	elif whichtype == "float":
		return float(val)
	elif whichtype == "bool":
		if val.lower() == "false" or val == "0":
			return False
		elif val.lower() == "true" or val == "1":
			return True
		else:
			raise ValueError()

def getInput(parameter, default, inputList, whichtype="float"):
	"""
	Gets the input by parsing the inputList.	
	"""
	for line in inputList:
		#a=re.fullmatch(paramName+"\s*=\s*([+-]?[0-9]+(?:\.?[0-9]*(?:[eE][+-]?[0-9]+)?)?)", line.strip())
		a=re.fullmatch(parameter+"\s*=\s*(.*)", line.strip())
		if a:
			val = a.group(1)

			try:
				return convertType(val, whichtype)
			except ValueError:
				raise ValueError(f"Value not understood: {parameter} = {val}")
	return default

def ExpectedInput(param, test):
	if not test:
		raise RuntimeError(f"Input for {param} not recognized!")


# parameters class holds all neccessary data for the calculation ##################################

class PARAMETERS:

	def __init__(self, inputFile):

		self.inputList = getInputList(inputFile)

		# these parameters should be the same for all SC islands:
		self.D = getInput("D", 1, self.inputList, "float")
		self.NSC = getInput("NSC", 1, self.inputList, "int")

		self.mode = getInput("mode", "homo_chain", self.inputList, "string")
		ExpectedInput("mode", self.mode == "homo_chain" or self.mode == "specify_each")

		self.chain, self.couplings, self.QDcount, self.SCcount = self.parseModel(self.inputList, self.D, self.NSC, self.mode)

		self.N = self.QDcount + (self.SCcount * self.NSC)

		#calculation parameters:
		self.parallel = getInput("parallel", False, self.inputList, "bool")

		self.band_level_shift = getInput("band_level_shift", True, self.inputList, "bool")
		self.flat_band = getInput("flat_band", False, self.inputList, "bool")
		
		self.nrange = getInput("nrange", 0, self.inputList, "int")
		self.refisn0 = getInput("refisn0", False, self.inputList, "bool")

		self.Eshift = self.getEshift(self.chain)
		self.nref = getInput("nref", self.defaultnref(self.chain), self.inputList, "int")

		self.subspace_set = getInput("subspace_set", "basic", self.inputList, "string") 	#setting which subspace to calculate
		
		self.excited_states = getInput("excited_states", 2, self.inputList, "int")			#how many states to calculate
		self.all_states = getInput("all_states", False, self.inputList, "bool")				#if this is true, calculate as many states as possible

		#postprocessing parameters

		self.print_vector = getInput("print_vector", False, self.inputList, "bool")			#whether to print the vectors of all calculated states
		self.print_vector_precision = getInput("print_vector_precision", 1e-3, self.inputList, "float")	#how prominent the contribution of a basis state has to be to be printed

		self.end_to_end_spin_correlations = getInput("end_to_end_spin_correlations", False, self.inputList, "bool")
		self.spin_correlation_matrix = getInput("spin_correlation_matrix", False, self.inputList, "bool")



	def parse_homo_chain(self, chainString, inputList, D, NSC):	

		chain, couplings = [], []
		QDcount, SCcount = 0, 0
		countLevels = 0
		chainParts = chainString.split("-")
		for i in range(len(chainParts)):
						
			chainPart = chainParts[i]

			if chainPart == "QD":
				QDcount+=1

				level 	= countLevels
				countLevels+=1

				U 		= getInput("U", 10, inputList, "float")
				epsimp 	= getInput("epsimp", -U/2, inputList, "float")
				EZ_imp	= getInput("EZ_imp", 0, inputList, "float")

				qd = QD(level, U, epsimp, EZ_imp)
				chain.append(qd)

			elif chainPart == "SC":
				SCcount+=1

				levels = [countLevels + i for i in range(NSC)]
				countLevels += NSC

				alpha	= getInput("alpha", 0, inputList, "float")	 
				Ec		= getInput("Ec", 0, inputList, "float")
				n0		= getInput("n0", 0, inputList, "float")
				EZ		= getInput("EZ", 0, inputList, "float")

				sc = SC(levels, NSC, D, alpha, Ec, n0, EZ)
				chain.append(sc)

			if i < len(chainParts) - 1:
				if chainParts[i] == "QD" and chainParts[i+1] == "QD":
					t = getInput("t", 0, inputList, "float")
					V = getInput("V", 0, inputList, "float")
					hop = HOP(t, V)
					couplings.append(t)
				else:
					Gamma = getInput("Gamma", 0, inputList, "float")			
					V = getInput("V", 0, inputList, "float")			
					hyb = HYB(Gamma, V)
					couplings.append(hyb)	
		return chain, couplings, QDcount, SCcount


	def parse_specified_chain(self, chainString, inputList, D, NSC):

		chain, couplings = [], []
		QDcount, SCcount = 0, 0
		countLevels = 0
		chainParts = chainString.split("-")
		for i in range(len(chainParts)):
						
			chainPart = chainParts[i]

			if chainPart == "QD":
				QDcount+=1

				level 	= countLevels
				countLevels+=1

				U 		= getInput(f"U{QDcount}", 10, inputList, "float")
				epsimp 	= getInput(f"epsimp{QDcount}", -U/2, inputList, "float")
				EZ_imp	= getInput(f"EZ_imp{QDcount}", 0, inputList, "float")

				qd = QD(level, U, epsimp, EZ_imp)
				chain.append(qd)

			elif chainPart == "SC":
				SCcount+=1

				levels = [countLevels + i for i in range(NSC)]
				countLevels += NSC

				alpha	= getInput(f"alpha{SCcount}", 0, inputList, "float")	 
				Ec		= getInput(f"Ec{SCcount}", 0, inputList, "float")
				n0		= getInput(f"n0{SCcount}", 0, inputList, "float")
				EZ		= getInput(f"EZ{SCcount}", 0, inputList, "float")

				sc = SC(levels, NSC, D, alpha, Ec, n0, EZ)
				chain.append(sc)

			if i < len(chainParts) - 1:
				if chainParts[i] == "QD" and chainParts[i+1] == "QD":
					t = getInput(f"t{i+1}", 0, inputList, "float")
					V = getInput(f"V{i+1}", 0, inputList, "float")
					hop = HOP(t, V)
					couplings.append(t)

				else:
					Gamma = getInput(f"Gamma{i+1}", 0, inputList, "float")			
					V = getInput(f"V{i+1}", 0, inputList, "float")			
					hyb = HYB(Gamma, V)
					couplings.append(hyb)		

		return chain, couplings, QDcount, SCcount
		


	def parseModel(self, inputList, D, NSC, mode):

		# get the string description of a chain
		chainString = getInput("model", "", inputList, "string")
		if chainString == "":
			raise Exception("Model not given!")

		if mode == "homo_chain":
			chain, couplings, QDcount, SCcount = self.parse_homo_chain(chainString, inputList, D, NSC)
		elif mode == "specify_each":
			chain, couplings, QDcount, SCcount = self.parse_specified_chain(chainString, inputList, D, NSC)


		print(f"The chain is: {chainString}")
		print(f"It has {QDcount} impurities and {SCcount} SC islands.")

		return chain, couplings, QDcount, SCcount

	def getEshift(self, chain):
		Eshift = 0
		for part in chain:
			if isinstance(part, QD):
				Eshift += part.U/2
			elif isinstance(part, SC):
				Eshift += part.Ec * (part.n0**2)
		return Eshift		

	def defaultnref(self, chain):
		default = 0
		for part in chain:
			if isinstance(part, QD):
				default += part.nu
			elif isinstance(part, SC):
				default += part.n0
		return int(default)

	def get_num_of_states(self, basis):
		num_of_states = self.excited_states
		if self.all_states:
			num_of_states = basis.size() - 1 	
		return num_of_states
		


