
import numpy as np

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

    print(f"A: {A}")
    print(f"B: {B}")
    print(f"distances_manual: {distances_manual}")
    print(f"distances_numpy: {distances_numpy}")
    print(f"error: {error}")



class KNN:
    def __init__(self, n, d):
        """
        初始化 KNN 类
        :param n: 向量数量
        :param d: 向量维度
        """
        self.n = n
        self.d = d
        self.data = self._generate_vectors()

    def _generate_vectors(self):
        """
        随机生成 n 个 d 维向量
        :return: 生成的向量
        """
        return np.random.rand(self.n, self.d)

    def querB(self, querB_vector, k=1):
        """
        查询最近邻
        :param querB_vector: 查询向量
        :param k: 最近邻数量
        :return: 最近邻的索引
        """
        distances = pairwise_distances(self.data, querB_vector.reshape(1, -1))
        nearest_indices = np.argsort(distances.flatten())[:k]
        nearest_vectors = self.data[nearest_indices]  # 获取最近邻向量
        return nearest_indices, nearest_vectors

# 示例用法
if __name__ == "__main__":
    # knn = KNN(n=100, d=10)  # 生成 100 个 10 维向量
    # querB_vector = np.random.rand(10)  # 随机生成一个查询向量
    # print("查询向量:", querB_vector)
    # nearest_neighbors, nearest_vectors = knn.querB(querB_vector, k=5)  # 查询 5 个最近邻
    # print("最近邻索引:", nearest_neighbors)
    # print("最近邻向量:", nearest_vectors)
    
    pairwise_distances_test(10)