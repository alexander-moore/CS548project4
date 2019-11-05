# cluster.py
import sklearn.metrics as skm

def evaluate(true_labs, pred_labs, X, labels): # evaluate WRT income, sex

	ajd_rand = skm.adjusted_rand_score(true_labs, pred_labs)
	norm_info = skm.normalized_mutual_info_score(true_labs, pred_labs)
	adj_info = skm.adjusted_mutual_info_score(true_labs, pred_labs)
	homog = skm.homogeneity_score(true_labs, pred_labs)
	complete = skm.completeness_score(true_labs, pred_labs)
	v_measure = skm.v_measure_score(true_labs, pred_labs)
	silhuoette = skm.silhuoette_score(X, labels) # what ?
	cont_mat = skm.cluster.contingency_matrix(true_labs, pred_labs)

	return [adj_rand, norm_info, adj_info, homog, complete, v_measure, silhuoette], cont_mat

# sklearn.manifold.MDS

def run_biggggg(data, target = 'sex', ):

	# K-means clustering:

	# Heirarchical Clustering:

	# DBSCAN: