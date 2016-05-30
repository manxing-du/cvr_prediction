from sklearn.cluster import KMeans

def Run_Kmeans(mtx_train,mtx_test, num_clusters):
    est = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto',
                 verbose=0, random_state=10, copy_x=True, n_jobs=-2)
    kmeans_model = est.fit(mtx_train)
    cluster_train = kmeans_model.predict(mtx_train)
    cluster_test = kmeans_model.predict(mtx_test)
    return cluster_train, cluster_test
