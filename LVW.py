import random
import numpy as np
from collections import defaultdict


def incon_check(data, target):
    already_checked = np.zeros(data.shape[0])
    inconsistency_count = 0
    for i in range(data.shape[0]):
        if already_checked[i] == 1:
            continue
        counts = defaultdict(int)
        for j in range(i + 1, data.shape[0]):
            equal = True
            for k in range(data.shape[1]):
                if data.iloc[i, k] != data.iloc[j, k]:
                    equal = False
            if equal and target[i] != target[j]:
                if counts[target[i]] == 0:
                    counts[target[i]] = 1
                counts[target[j]] += 1
                already_checked[j] = 1
        largest_class = 0
        sum_v = 0
        for _, v in counts.items():
            sum_v += v
            if v > largest_class:
                largest_class = v
        inconsistency_count += sum_v - largest_class
    return inconsistency_count / data.size


def lvw(data, target, gamma=0., max_tries=100000, random_state=0):
    random.seed(random_state)
    n = data.shape[1]
    c_best = n
    s_best = None
    for _ in range(max_tries):
        c = random.randrange(1, n)
        indices = random.sample(range(0, n-1), c)
        s = data.iloc[:, indices]
        if c < c_best:
            if incon_check(s, target) <= gamma:
                s_best = s
                c_best = c
        elif c == c_best and incon_check(s, target) <= gamma:
            print(s.columns)
    return s_best.columns.values

