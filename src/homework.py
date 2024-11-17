import numpy as np
from sklearn.cluster import KMeans
from kneed import KneeLocator
import warnings
warnings.filterwarnings('ignore')
import hnswlib
import time


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
        self.index = self.create_hnsw_index()

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

    def create_hnsw_index(self):
        # 创建 HNSW 索引
        index = hnswlib.Index(space='l2', dim=self.d)  # 使用 L2 距离
        index.init_index(max_elements=self.n, ef_construction=200, M=16)  # 初始化索引
        index.add_items(self.data, np.arange(self.n))  # 添加数据
        index.set_ef(50)  # 设置查询时的 ef 值
        return index


    def knn_search(self, query_vectors):
        """
        针对一组查询向量返回最近邻
        :param query_vectors: 查询向量，形状为 (m, d)
        :return: 最近邻的索引和向量
        """
        nearest_indices_list = []
        nearest_vectors_list = []

        for query_vector in query_vectors:
            indices, distances = self.index.knn_query(query_vector, k=self.k)  # 使用 HNSW 查询最近邻
            nearest_vectors = self.data[indices.flatten()]  # 获取最近邻向量
            
            nearest_indices_list.append(indices.flatten())
            nearest_vectors_list.append(nearest_vectors)

        return nearest_indices_list, nearest_vectors_list
    
def test_knn_functionality():
    """
    功能测试：验证 KNN 类的正确性
    """
    np.random.seed(123)  # 设置随机种子以确保可重复性
    n = 100  # 数据库中的向量数量
    d = 10   # 向量维度
    k = 3    # 最近邻数量

    knn = KNN(n=n, d=d, k=k)  # 创建 KNN 实例
    query_vectors = np.random.rand(5, d)  # 随机生成 5 个查询向量

    nearest_neighbors, nearest_vectors = knn.knn_search(query_vectors)  # 查询每个向量的最近邻
    print("*"*20,"The functional test of task2","*"*20)
    for i in range(len(query_vectors)):
        print(f"查询向量 {i} 的最近邻索引:", nearest_neighbors[i])
        print(f"查询向量 {i} 的最近邻向量:", nearest_vectors[i])
               
    # 验证返回的最近邻索引和向量的形状
    assert len(nearest_neighbors) == len(query_vectors), "最近邻索引的数量不正确"
    assert len(nearest_vectors) == len(query_vectors), "最近邻向量的数量不正确"

    for i in range(len(query_vectors)):
        # 检查返回的最近邻索引数量
        if len(nearest_neighbors[i]) < k:
            print(f"警告: 查询向量 {i} 的最近邻索引数量少于 k ({k})，实际数量为 {len(nearest_neighbors[i])}")
        else:
            assert len(nearest_neighbors[i]) == k, f"查询向量 {i} 的最近邻索引数量不正确"
        
        assert nearest_vectors[i].shape[1] == d, f"查询向量 {i} 的最近邻向量维度不正确"

    print("功能测试通过！")
    print("*"*70) 


def test_knn_performance():
    """
    性能测试：测量 KNN 查询的执行时间
    """
    np.random.seed(123)  # 设置随机种子以确保可重复性
    n = 10000  # 数据库中的向量数量
    d = 10     # 向量维度
    k = 5      # 最近邻数量

    knn = KNN(n=n, d=d, k=k)  # 创建 KNN 实例
    query_vectors = np.random.rand(100, d)  # 随机生成 100 个查询向量

    start_time = time.time()  # 记录开始时间
    nearest_neighbors, nearest_vectors = knn.knn_search(query_vectors)  # 查询每个向量的最近邻
    end_time = time.time()  # 记录结束时间

    print(f"性能测试：查询 100 个向量的最近邻耗时 {end_time - start_time:.4f} 秒")
    print("*"*70)

# 示例用法
if __name__ == "__main__":
    # task1
    # pairwise_distances_test(10)
    # task2
    test_knn_functionality()
    
    test_knn_performance()