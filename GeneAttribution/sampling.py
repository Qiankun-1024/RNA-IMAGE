import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
import numpy as np

from utils import Wasserstein_Distance, gene2img, generate_distribution


def cloud_point_sample(data, label, percent, dtype):
    
    npoint = round(data.shape[0] * percent)
    seeds = np.zeros(npoint)
    distance = np.ones(data.shape[0]) * 1e10

    import random
    random.seed(10)
    seed = random.randint(0, data.shape[0])
    
    means, logvars = generate_distribution(data, dtype=dtype, batch_size=64)
    print(means)
    
    for i in range(npoint):
        seeds[i] = seed
        d = one2other_distance(means, logvars, seed)
        mask = d < distance
        distance[mask] = d[mask]

        seed = np.argmax(distance)

    sample_data = data.iloc[seeds]
    sample_label = label.iloc[seeds]
    return sample_data, sample_label


def one2other_distance(means, logvars, idx):
    d = Wasserstein_Distance(means, logvars, means.iloc[idx], logvars.iloc[idx])
    return d


def random_sample(data, label, percent):
   
    import random
    random.seed(10)
    seed = random.randint(0, data.shape[0])
    
    sample_label = label.groupby('cancer_type').sample(frac=percent)
    sample_data = data.loc[sample_label.index]

    return sample_data, sample_label
