import torch
import torch.nn


class kmeans_cmm_layer(torch.nn.Module):
	def __init__(self, embeddign_dim: int, num_classes: int, return_gmm: bool = True, return_cmm: bool = True, return_cluster_dist: bool = True):
		super().__init__()

		self.dim = embeddign_dim
		self.num_classes = num_classes
		self.centers = torch.nn.Parameter(torch.rand(size=(self.num_classes, self.dim), requires_grad=True))

		self.return_gmm = return_gmm
		self.return_cmm = return_cmm
		self.return_cluster_dist = return_cluster_dist

	def forward(self, x):
		batch = x.size()[0]  # batch size

		# ---
		# Calculate distance to cluster centers
		# ---
		# Upsample the x-data to [batch, num_classes, dim]
		x_rep = x.unsqueeze(1).repeat(1, self.num_classes, 1)

		# Upsample the clusters to [batch, num_classes, dim]
		centers_rep = self.centers.unsqueeze(0).repeat(batch, 1, 1)

		# Subtract to get diff of [batch, num_classes, dim]
		diff = x_rep - centers_rep
		
		# Obtain the square distance to each cluster
		#  of size [batch, dim]
		dist_sq = diff*diff
		dist_sq = torch.sum(dist_sq,2)

		#   GMM
		# dist_sq = (x-mu) Sigma_inv (x-mu)T
		#   K-means
		# dist_sq = (x-mu) dot (x-mu)

		# Obtain the exponents
		expo_kmeans = -0.5 * dist_sq
		a = torch.add(dist_sq, 1)
		b = -((1 + self.dim) / 2)
		expo_cmm = torch.pow(a, b)

		# Obtain the "safe" (numerically stable) versions of the
		#  exponents.  These "safe" exponents produce fake numer and denom
		#  but guarantee that resp = fake_numer / fake_denom = numer / denom
		#  where fake_numer and fake_denom are numerically stable
		expo_safe_off_kmeans, _ = torch.max(expo_kmeans, dim=-1, keepdim=True)
		expo_safe_kmeans = expo_kmeans - expo_safe_off_kmeans  # use broadcast instead of the repeat
		expo_safe_off_cmm, _ = torch.max(expo_cmm, dim=-1, keepdim=True)
		expo_safe_cmm = expo_cmm - expo_safe_off_cmm  # use broadcast instead of the repeat
		
		# Calculate the responsibilities kmeans
		numer_safe_kmeans = torch.exp(expo_safe_kmeans)
		denom_safe_kmeans = torch.sum(numer_safe_kmeans, 1, keepdim=True)
		resp_kmeans = numer_safe_kmeans / denom_safe_kmeans  # use broadcast

		# Calculate the responsibilities cmm
		numer_safe_cmm = torch.exp(expo_safe_cmm)
		denom_safe_cmm = torch.sum(numer_safe_cmm, 1, keepdim=True)
		resp_cmm = numer_safe_cmm / denom_safe_cmm  # use broadcast

		# # Build the kmeans resp
		# dist_sq_kmeans = torch.sum(diff * diff, dim=2)
		# # Obtain the exponents
		# expo = -0.5 * dist_sq_kmeans
		#
		# # Obtain the "safe" (numerically stable) versions of the
		# #  exponents.  These "safe" exponents produce fake numer and denom
		# #  but guarantee that resp = fake_numer / fake_denom = numer / denom
		# #  where fake_numer and fake_denom are numerically stable
		# expo_safe_off = torch.mean(expo, dim=-1, keepdim=True)
		# expo_safe = expo - expo_safe_off
		#
		# # Calculate the responsibilities
		# numer_safe = torch.exp(expo_safe)
		# denom_safe = torch.sum(numer_safe, 1, keepdim=True)
		# resp_kmeans = numer_safe / denom_safe

		# argmax resp to assign to cluster
		# optimize CE over resp + L2 loss

		#
		# Vectorized version of cluster_dist
		#
		
		# Obtain cluster assignment from dist_sq directly
		cluster_assignment = torch.argmin(dist_sq, dim=-1)

		# Use one-hot encoding trick to extract the dist_sq
		cluster_assignment_onehot = torch.nn.functional.one_hot(cluster_assignment, dist_sq.shape[1])
		cluster_dist_sq_onehot	= cluster_assignment_onehot * dist_sq
		cluster_dist_sq		   = torch.sum(cluster_dist_sq_onehot, dim=-1)

		# Take square root of dist_sq to get L2 norm
		cluster_dist = torch.sqrt(cluster_dist_sq)

		resp_kmeans = torch.clip(resp_kmeans, min=1e-8)
		resp_cmm = torch.clip(resp_cmm, min=1e-8)
		resp_kmeans = torch.log(resp_kmeans)
		resp_cmm = torch.log(resp_cmm)

#		print("resp_cmm")
#		print(resp_cmm)
#		input("")

		outputs = list()
		if self.return_gmm:
			outputs.append(resp_kmeans)
		if self.return_cmm:
			outputs.append(resp_cmm)
		if self.return_cluster_dist:
			outputs.append(cluster_dist)

		return outputs
	