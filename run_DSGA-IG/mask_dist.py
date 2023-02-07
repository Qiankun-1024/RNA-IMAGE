import os
import sys
path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)

import pandas as pd
from utils import normalize_feature
from tensorflow.python.eager.backprop import make_attr

from GeneAttribution import ComparisonAnalysis
from GeneAttribution.comparisonIG import ImportantGene, compute_mask_distance
from utils import gene2img, load_model

#sample data
label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv',
                   sep='\t', index_col=0)
data = pd.read_csv('./data/TCGA_GTEx/log2_x_data.tsv',
                   sep='\t', index_col=0)
label = label.sample(n=6, random_state=42)
data = data.loc[label.index]
data = gene2img(data, dtype='log2TPM')

model = load_model(only_encoder=True)

mask_dist = []
id_list = []
for i in range(len(label)-1):
    for j in range((i+1),len(label)):
        id_list.append(list(label.iloc[[i,j]].index))
        tmp_dist = compute_mask_distance(model, data[i], data[j], 20)
        mask_dist.append(tmp_dist)
mask_dist = pd.DataFrame(mask_dist)
id_list = pd.DataFrame(id_list)
df = pd.concat([id_list, mask_dist], axis=1)
df.to_csv('./data/output/attrs/mask_dist.tsv',
          sep='\t')