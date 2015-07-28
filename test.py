from preprocess import load_my_data, feat_extraction
from visualize import plot_2D_transformation, pairwise_scatter, plot_2D_clustering
from sklearn.cluster import KMeans

__author__ = 'matteo'

# initialization
path = "iris-data.csv"

# load data
(X,y) = load_my_data(path)

# plot original data 2 factor scatterplots
pairwise_scatter(X,y)

# supervised (LDA) and unsupervised(PCA) feature extraction
num_feat = 2
(X_pca, X_lda) = feat_extraction(X,y,num_feat)

# plotting 2D transformed data
plot_2D_transformation(X_pca)
plot_2D_transformation(X_lda)

# cluster pca transformed data
clstr = KMeans(n_clusters=3)
clstr.fit(X_pca)
labels = clstr.labels_
plot_2D_clustering(X_pca, labels)

# plot lda transformed data
clstr.fit(X_lda)
labels = clstr.labels_
plot_2D_clustering(X_lda, labels)
