import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')

def pairwise_distances(A, B):
    '''
    return the matriA of distances between the rows of A and the rows of B 
    '''
    ret = 0
    for i in range(A.shape[0]):
        ret += np.square(A[i]-B[i])
    distance = np.sqrt(ret)
    return distance

def pairwise_distances_numpy(A, B):
    '''
    directly call the method of calculating vector distance in numpy
    '''
    return np.linalg.norm(A - B)

def pairwise_distances_test(d):
    '''
    test the function of pairwise_distances(A, B)
    '''
    A = np.random.rand(d)
    B = np.random.rand(d)

    distances_manual = pairwise_distances(A, B)
    distances_numpy = pairwise_distances_numpy(A, B)

    error = abs(distances_manual - distances_numpy)
    print("The functional test of task1")
    print("*"*20,"The functional test of task1","*"*20) 
    print(f"A: {A}")
    print(f"B: {B}")
    print(f"distances_manual: {distances_manual}")
    print(f"distances_numpy: {distances_numpy}")
    print(f"error: {error}")
    print("*"*70) 



class KNN:
    def __init__(self, n, d, k=1):
        """
        Initialize KNN class
        :param n: number of vectors
        :param d: vector dimension
        """
        self.n = n
        self.d = d
        self.k = k
        self.data = self.generate_vectors()

    def generate_vectors(self, n_clusters=4):
        """
        使用多元正态分布生成 n 个 d 维向量
        :param n_clusters: 聚类数量
        :return: 生成的向量
        """
        means = np.random.rand(n_clusters, self.d)  # 随机生成聚类中心
        covariances = [np.eye(self.d) for _ in range(n_clusters)]  # 使用单位矩阵作为协方差矩阵
        
        vectors = []
        for i in range(self.n):
            cluster = np.random.choice(n_clusters)  # 随机选择一个聚类
            vector = np.random.multivariate_normal(means[cluster], covariances[cluster])  # 生成多元正态分布向量
            vectors.append(vector)

        return np.array(vectors)

    def get_best_k(self, vectors):
        sse = []
        k_values = range(2, int(np.log2(len(vectors))) + 1)
        
        for k in k_values:
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(vectors)
            sse.append(kmeans.inertia_)
        
        knee_locator = KneeLocator(k_values, sse, curve='convex', direction='decreasing')
        return knee_locator.elbow

    def cluster(self, vectors):
        best_k = self.get_best_k(vectors)
        kmeans = KMeans(n_clusters=best_k)
        kmeans.fit(vectors)
        return kmeans.labels_


    def knn_search(self, query_vectors):
        """
        针对一组查询向量返回最近邻
        :param query_vectors: 查询向量，形状为 (m, d)
        :return: 最近邻的索引和向量
        """
        nearest_indices_list = []
        nearest_vectors_list = []

        for query_vector in query_vectors:
            distances = np.array([pairwise_distances(self.data[i], query_vector) for i in range(self.n)])  # 计算距离
            nearest_indices = np.argsort(distances)[:self.k]  # 获取最近邻索引
            nearest_vectors = self.data[nearest_indices]  # 获取最近邻向量
            
            nearest_indices_list.append(nearest_indices)
            nearest_vectors_list.append(nearest_vectors)

        return nearest_indices_list, nearest_vectors_list
    
    

# 示例用法
if __name__ == "__main__":
    # task1
    # pairwise_distances_test(10)
    # task2
    knn = KNN(n=100, d=10, k=3)  # 创建 KNN 实例，设置 k=3
    query_vectors = np.random.rand(5, 10)  # 随机生成 5 个查询向量
    nearest_neighbors, nearest_vectors = knn.knn_search(query_vectors)  # 查询每个向量的最近邻

    for i in range(len(query_vectors)):
        print(f"查询向量 {i} 的最近邻索引:", nearest_neighbors[i])
        print(f"查询向量 {i} 的最近邻向量:", nearest_vectors[i])