# SMOTE算法及其python实现
import random
from sklearn.neighbors import NearestNeighbors
import numpy as np


class Smote:
    def __init__(self, samples, N=10, k=5):
        # N:采样倍率
        self.N = N
        # k:近邻数
        self.k = k
        self.samples = samples

    def over_sampling(self):
        neighbors = NearestNeighbors(n_neighbors=self.k).fit(self.samples)
        print('neighbors', neighbors)
        synthetic = []
        for i in range(len(self.samples)):
            print('samples', self.samples[i])
            nnarray = neighbors.kneighbors(self.samples[i].reshape((1, -1)), return_distance=False)[0]  # Finds the K-neighbors of a point.
            print('nna', nnarray)
            synthetic = self._populate(i, nnarray, synthetic)
        return synthetic

    # for each minority class sample i ,choose N of the k nearest neighbors and generate N synthetic samples.
    def _populate(self, i, nnarray, synthetic):
        for j in range(self.N):
            nn = random.randint(0, self.k - 1)  # 包括end
            dif = self.samples[nnarray[nn]] - self.samples[i]
            gap = random.random()
            synthetic.append(self.samples[i] + gap * dif)
            return synthetic


if __name__ == '__main__':
    a = np.array([[1, 2, 3], [4, 5, 6], [2, 3, 1], [2, 1, 2], [2, 3, 4], [2, 3, 4]])
    s = Smote(a, N=5, k=3)
    aa = s.over_sampling()
    print(aa)
