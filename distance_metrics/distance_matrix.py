import os
import pandas as pd
from scipy import spatial
from metrics import *


def distance_matrix(mean, logvar):
    condensed_distance = []
    n = len(mean)
    for i in range(n):
        for j in range(n):
            if i < j:
                A_mean = mean.iloc[i].values
                A_logvar = logvar.iloc[i].values
                B_mean = mean.iloc[j].values
                B_logvar = logvar.iloc[j].values
                d = Wasserstein_Distance(A_mean, A_logvar, B_mean, B_logvar)
                condensed_distance.append(d)
    squared_distance= spatial.distance.squareform(condensed_distance)
    distance_matrix = pd.DataFrame(squared_distance,
                                   index=mean.index,
                                   columns=mean.index)
    return distance_matrix

if __name__ == '__main__':
    label = pd.read_csv('./data/TCGA_GTEx/all_label.tsv', index_col=0, sep='\t')
    label = label.drop(['batch'], axis=1)
    mean = pd.read_csv('./data/output/mean.tsv', index_col=0, sep='\t')
    logvar = pd.read_csv('./data/output/logvar.tsv', index_col=0, sep='\t')
    
    ct_list = ['BRCA', 'KIRP', 'CESC']
    for ct in ct_list:
        label_ = label[label['cancer_type']==ct]
        mean_ = mean.loc[label_.index]
        logvar_ = logvar.loc[label_.index]
        dist_mat = distance_matrix(mean_, logvar_)
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ct + '_distance_matrix.csv'))
        )

