import numpy as np
import sys


maxK=8

class GaussMoments:

	#---------------------------------------------------------------
	#---------------------------------------------------------------
	# Instance variables
	#  self.D                   int  (input)       number of dimensions
	#  self.K                   int  (input)       order of moments
	#  self.moment_weights      [D,D,..,D] K dims  weighting function for loss
	#  self.count_diags         [D,D,..,D] K dims  number of diagnoals the element intersects
	#  self.uni_gauss_moments   [K]                moments of a univariate standard normal distribution
	#  self.joint_gauss_moments [D,D,..,D] K dims  moments of a multivariate joint independent standard normal distribution
	#---------------------------------------------------------------
	#---------------------------------------------------------------

	def __init__(self,D,K):
		maxK = 8
		if (K>maxK):
			print("Error Moments.__init__(D=%d,K=%d)" % (D,K))
			print("  K %d > maxK %d" %(K,maxK))
			print("  large K not supported")
			sys.exit(1)

		#--------------------------------------------------
		# Calculate the number of occurances
		#  of each type for the skew matrix
		#--------------------------------------------------
		n_diags = K*(K-1) // 2
		# print(n_diags)

		# The list of dimensions of the tensor
		dims=[1]*maxK
		for i in range((maxK-K),maxK):
		  dims[i] = D
		# print(dims)

		# Create a numpy array with those dimensions
		count_diags = np.zeros(dims, dtype=np.int32)

		# For every pair of diagonals
		for b in range(K):
			for a in range(b):
			
				# for every diagonal entry
				for c in range(D):
			
					# start and end indices
					s = [0]*maxK
					e = dims.copy()
					s[maxK-a-1] = c
					e[maxK-a-1] = c+1
					s[maxK-b-1] = c
					e[maxK-b-1] = c+1
										
					# add one across all of the dimensions
					count_diags[s[0]:e[0], s[1]:e[1], s[2]:e[2], s[3]:e[3], s[4]:e[4], s[5]:e[5], s[6]:e[6], s[7]:e[7]] += 1

		count_diags = count_diags.reshape([D]*K)

		#print("count_diags")
		#print(count_diags)

		#
		# How many entries per diag
		#
		count_diags_flat = count_diags.reshape([D**K])
		cell_per_diag = [0]*(n_diags+1)
		for x in count_diags_flat:
			cell_per_diag[x]+=1


		#print("cell_per_diag")
		#print(cell_per_diag)

		#
		# How many non-zero diags
		#
		n_nonzero_diag = 0
		for g in range(n_diags+1):
			if (cell_per_diag[g]!=0):
				n_nonzero_diag += 1
				
		#print("n_nonzero_diag", n_nonzero_diag)

		#--------------------------------------------------
		# Calculate the weights for the method of moments
		#  penalty function on a per-element basis
		#--------------------------------------------------
		weight_per_diag = [0.0]*(n_diags+1)
		for g in range(n_diags+1):
			if (cell_per_diag[g]!=0):
				weight_per_diag[g] = 1.0 / (cell_per_diag[g] * n_nonzero_diag)

		#print("weight_per_diag")
		#print(weight_per_diag)
		#input("press enter")

		#
		# Calculate the weights of the moments
		#
		idx = np.asarray(list(range(D**K)))
		moment_weights = np.asarray(weight_per_diag)[count_diags_flat[idx]]

		# moment_weights = np.zeros([D ** K], dtype=np.float32)
		# total_weights = 0.0
		# for i in range(D**K):
		# 	moment_weights[i] = weight_per_diag[count_diags_flat[i]]
		# 	total_weights += moment_weights[i]

		moment_weights = moment_weights.reshape([D]*K)

		#--------------------------------------------------
		# Calculate the method of moments set values for the
		#  joint standard normal distribution
		#--------------------------------------------------

		#
		# Univariate Gaussian Moments
		#
		uni_gauss_moments = [0]*K
		double_factorial = 1.0
		for p in range(K):
			if (p%2)==0:
				uni_gauss_moments[p] = 0.0
			else:
				double_factorial *= p
				uni_gauss_moments[p] = double_factorial

		#print("uni_gauss_moments")
		#print(uni_gauss_moments)
		#input("press enter")

		#
		# Joint Gaussian Moments
		#
		counter = [0]*maxK
		joint_gauss_moments = np.zeros([D**K], dtype=np.float32)
		for i in range(D**K):

			# loop through and calculate the joint moment
			joint_moment = 1.0
			
			idx_seen = [False]*K
			for idx in range(K):
				if (idx_seen[idx]==False):
				
					# Which dimension are we counting
					dim = counter[maxK-idx-1]
					idx_seen[idx] = True
					dim_count = 1

					# See if that dimension is multiplied by itself in this entry
					for idx2 in range(idx+1,K):
						if (idx_seen[idx2]==False):
							dim2 = counter[maxK-idx2-1]
							if (dim2==dim):
								dim_count+=1
								idx_seen[idx2] = True

					# What is the univariate moment
					uni_moment = uni_gauss_moments[dim_count-1]
					
					# Multiply the joint moment by the univariate moment
					joint_moment *= uni_moment

			# Record the joint moment
			joint_gauss_moments[i] = joint_moment

			# increment counter
			digit = maxK-1
			counter[digit]+=1
			while digit>=0 and counter[digit]>=dims[digit]:
				counter[digit]=0
				digit-=1
				if (digit>=0):
					counter[digit]+=1

		joint_gauss_moments = joint_gauss_moments.reshape([D]*K)

		#print("joint_gauss_moments")
		#print(joint_gauss_moments)

		#------------------------
		# Pack the results into the instance variables
		#------------------------
		self.D                   = D                   # int  (input)       number of dimensions
		self.K                   = K                   # int  (input)       order of moments
		self.moment_weights      = moment_weights      # [D,D,..,D] K dims  weighting function for loss
		self.count_diags         = count_diags         # [D,D,..,D] K dims  number of diagnoals the element intersects
		self.uni_gauss_moments   = uni_gauss_moments   # [K]                moments of a univariate standard normal distribution
		self.joint_gauss_moments = joint_gauss_moments # [D,D,..,D] K dims  moments of a multivariate joint independent standard normal distribution



