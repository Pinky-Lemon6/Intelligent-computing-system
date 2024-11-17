import numpy as np
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')
import hnswlib
import time

# task1
def pairwise_distances(A, B):
    """
    return the matriA of distances between the rows of A and the rows of B 
    """
    ret = 0
    for i in range(A.shape[0]):
        ret += np.square(A[i]-B[i])
    distance = np.sqrt(ret)
    return distance

def pairwise_distances_numpy(A, B):
    """
    directly call the method of calculating vector distance in numpy
    """
    return np.linalg.norm(A - B)

def pairwise_distances_test(d):
    """
    test the function of pairwise_distances(A, B)
    """
    A = np.random.rand(d)
    B = np.random.rand(d)
    

    distances_manual = pairwise_distances(A, B)
    distances_numpy = pairwise_distances_numpy(A, B)

    error = abs(distances_manual - distances_numpy)
    print("*"*20,"The functional test of task1","*"*20) 
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"distances_manual: {distances_manual}")
    print(f"distances_numpy: {distances_numpy}")
    print(f"error: {error}")
    print("*"*70) 

# task2
class KNN:
    def __init__(self, n, d, k=1):
        """
        Initialize KNN class
        :param n: number of vectors
        :param d: vector dimension
        :param k: number of neighbors
        """
        self.n = n
        self.d = d
        self.k = k
        self.data = self.generate_vectors()
        self.index = self.create_hnsw_index()

    def generate_vectors(self, n_clusters=4):
        """
        Generate n d-dimensional vectors using multivariate normal distribution
        :param n_clusters: number of clusters
        :return: generated vector
        """
        means = np.random.rand(n_clusters, self.d)  # 随机生成聚类中心
        covariances = [np.eye(self.d) for _ in range(n_clusters)]  # 使用单位矩阵作为协方差矩阵
        
        vectors = []
        for i in range(self.n):
            cluster = np.random.choice(n_clusters)  # 随机选择一个聚类
            vector = np.random.multivariate_normal(means[cluster], covariances[cluster])  # 生成多元正态分布向量
            vectors.append(vector)

        return np.array(vectors)

    def create_hnsw_index(self):
        # 创建 HNSW 索引
        index = hnswlib.Index(space='l2', dim=self.d)  # 使用 L2 距离
        index.init_index(max_elements=self.n, ef_construction=200, M=16)  # 初始化索引
        index.add_items(self.data, np.arange(self.n))  # 添加数据
        index.set_ef(50)  # 设置查询时的 ef 值
        return index

    def knn_search(self, query_vectors):
        """
        Returns the nearest neighbor for a set of query vectors
        :param query_vectors: query vectors, shape (m, d)
        :return: the index and vector of the nearest neighbor
        """
        nearest_indices_list = []
        nearest_vectors_list = []

        for query_vector in query_vectors:
            indices, distances = self.index.knn_query(query_vector, k=self.k)  # 使用 HNSW 查询最近邻
            nearest_vectors = self.data[indices.flatten()]  # 获取最近邻向量
            
            nearest_indices_list.append(indices.flatten())
            nearest_vectors_list.append(nearest_vectors)

        return nearest_indices_list, nearest_vectors_list
    
    def kmeans_knn_search(self, query_vectors, n_clusters=5):
        """
        Use KMeans clustering followed by brute force search to find the nearest neighbors
        :param query_vectors: query vectors, shape (m, d)
        :param n_clusters: number of clusters
        :return: the index and vector of the nearest neighbors
        """
        # 进行 KMeans 聚类
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(self.data)
        
        nearest_indices_list = []
        nearest_vectors_list = []

        for query_vector in query_vectors:
            # 计算查询向量与每个聚类中心的距离
            distances_to_centers = np.linalg.norm(kmeans.cluster_centers_ - query_vector, axis=1)
            nearest_cluster_index = np.argmin(distances_to_centers)  # 找到最近的聚类中心

            # 在最近的聚类中进行暴力搜索
            cluster_indices = np.where(kmeans.labels_ == nearest_cluster_index)[0]
            cluster_vectors = self.data[cluster_indices]
            distances = np.array([pairwise_distances(cluster_vectors[i], query_vector) for i in range(len(cluster_vectors))])
            nearest_indices = np.argsort(distances)[:self.k]  # 获取最近邻索引
            
            nearest_indices_list.append(cluster_indices[nearest_indices])
            nearest_vectors_list.append(cluster_vectors[nearest_indices])

        return nearest_indices_list, nearest_vectors_list
    
def test_knn_functionality(n, d, k):
    """
    functional testing: verifying the correctness of the KNN class
    """
    np.random.seed(123)  
    knn = KNN(n=n, d=d, k=k)  
    query_vectors = np.random.rand(5, d)  

    nearest_neighbors, nearest_vectors = knn.knn_search(query_vectors)  
    print("*"*20,"The functional test of task2","*"*20)
    for i in range(len(query_vectors)):
        print(f"The nearest neighbor indices of vector {i}:", nearest_neighbors[i])
        print(f"The nearest neighbor vectors of vector {i}:", nearest_vectors[i])
               
    
    assert len(nearest_neighbors) == len(query_vectors), "Incorrect number of nearest neighbor indices!"
    assert len(nearest_vectors) == len(query_vectors), "Incorrect number of nearest neighbor vectors!"

    for i in range(len(query_vectors)):
        assert len(nearest_neighbors[i]) == k, f"Incorrect number of nearest neighbor indices for query vector {i}"
        
        assert nearest_vectors[i].shape[1] == d, f"The nearest neighbor vector dimensions for query vector {i} are incorrect"

    print("Functional test passed!")
    print("*"*70) 


def test_knn_performance(n, d, k):
    """
    Performance test: measure and compare the time it takes HNSW and Kmeans to query the knn of the same set of vectors
    """
    np.random.seed(123)  

    knn = KNN(n=n, d=d, k=k)  
    query_vectors = np.random.rand(100, d)  
    # HNSW
    start_time_1 = time.time()  
    nearest_neighbors_1, nearest_vectors_1 = knn.knn_search(query_vectors)  
    end_time_1 = time.time()  
    
    print("*"*20,"The perfomance test of task2","*"*20)
    print(f"HNSW query for the nearest neighbors of 100 vectors takes: {end_time_1 - start_time_1:.4f} seconds")
    # Kmeans
    start_time_2 = time.time()  
    nearest_neighbors_2, nearest_vectors_2 = knn.kmeans_knn_search(query_vectors)  
    end_time_2 = time.time()
    print(f"Kmeans query for the nearest neighbors of 100 vectors takes: {end_time_2 - start_time_2:.4f} seconds")
    print("*"*70)

# 示例用法
if __name__ == "__main__":
    n = 10000
    dim = 64 
    k = 3
    # task1
    # pairwise_distances_test(dim)
    # # task2
    # test_knn_functionality(n,dim,k)
    
    test_knn_performance(n,dim,k)