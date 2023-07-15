import random
import numpy as np
import math


class SelfAdaption(object):
    def __init__(self, label, sam_p):

        self.label = label
        self.sam_p = sam_p
        # self.high = label.shape[0]  # n = len(self.labels),不是label.shape[0]，弄混了调了很久

    # def subsample_idx(self, low, high, sample_size):
    #
    #     return np.random.randint(low, high, sample_size)

    def cut_adapt(self, step):

        sum_p, _ = self.adapt()
        h, w = self.label.shape
        ss_idx = []
        cut = [(i, j) for i in range(0, w-step+1, step) for j in range(0, h-step+1, step)]
        n_p = 0
        t = w//step

        for x, y in enumerate(cut):

            temp = self.label[y[0]:y[0] + step, y[1]:y[1] + step]
            n = temp.shape[0]*temp.shape[1]
            p_idx = []
            n_idx = []

            [p_idx.append(x//t*step*w+i*w+(x % t)*step+j) for i in range(temp.shape[0])
             for j in range(temp.shape[1]) if temp[i][j] == 255]
            [n_idx.append(x//t*step*w+i*w+(x % t)*step+j) for i in range(temp.shape[0])
             for j in range(temp.shape[1]) if temp[i][j] == 0]

            l_p = len(p_idx)
            l_n = len(n_idx)
            assert(l_p+l_n == n)
            n_p += l_p

            if l_p != 0:
                sample_p = math.ceil(self.sam_p*l_p/sum_p)
                p_idx = random.sample(p_idx, sample_p)
                sample_n = self.sam_p*l_p//sum_p*l_n//l_p
                n_idx = random.sample(n_idx, sample_n)
                # if sample_p != 0 or sample_n != 0:
                #     ss_idx.extend(p_idx)
                #     ss_idx.extend(n_idx)
                    # ss_idx = np.concatenate((ss_idx, p_idx, n_idx), axis=0)
            else:
                n_idx = random.sample(n_idx, 10)  # random.randint(0, 1)

            ss_idx.extend(p_idx)
            ss_idx.extend(n_idx)

        ss_idx = np.array(ss_idx)
        assert (n_p == sum_p)

        return ss_idx

    def adapt(self):

        labels = self.label.reshape(self.label.shape[0]*self.label.shape[1], 1)
        ss_idx = []
        n = len(labels)

        [ss_idx.append(i) for i in range(n) if labels[i] == 255]
        x = len(ss_idx)
        ss_idx = random.sample(ss_idx, self.sam_p)

        sample_size = round((n-x)*self.sam_p/x)
        ss_idx = np.append(np.array(ss_idx), np.random.randint(0, n, sample_size))

        return x, ss_idx
