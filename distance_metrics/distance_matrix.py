import os
import pandas as pd
from scipy import spatial
from metrics import *


def distance_matrix(z, mean, logvar, metric='Wasserstein'):
    condensed_distance = []
    n = len(mean)
    for i in range(n):
        for j in range(n):
            if i < j:
                x, y = z.iloc[i].values, z.iloc[j].values
                A_mean = mean.iloc[i].values
                A_logvar = logvar.iloc[i].values
                B_mean = mean.iloc[j].values
                B_logvar = logvar.iloc[j].values
                if metric == 'Wasserstein':
                    d = Wasserstein_Distance(A_mean, A_logvar, B_mean, B_logvar)
                elif metric == 'NPD':
                    eps_x = np.random.normal(size=(50,len(A_mean)))
                    x_posterior = eps_x * np.exp(A_logvar * .5) + A_mean
                    eps_y = np.random.normal(size=(50,len(B_mean)))
                    y_posterior = eps_y * np.exp(B_logvar * .5) + B_mean
                    d = npd(x, y, x_posterior, y_posterior)
                elif metric == 'Euclidean':
                    d = Euclidean_Distances(x, y)
                elif metric == 'Cosine':
                    d = Cosine_Distances(x, y)
                elif metric == 'Pearson':
                    d = Pearson_Distances(x, y)
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
    z = pd.read_csv('./data/output/zc.tsv', index_col=0, sep='\t')
    
    ct_list = ['BRCA', 'KIRP', 'KIRC']

    for ct in ct_list:
        label_ = label[label['cancer_type']==ct]
        z_ = z.loc[label_.index]
        mean_ = mean.loc[label_.index]
        logvar_ = logvar.loc[label_.index]
        dist_mat = distance_matrix(z_, mean_, logvar_)
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ct + '_distance_matrix.csv'))
        )

    ps_list = ['Kidney']
    for ps in ps_list:
        label_ = label[label['primary_site']==ps]
        z_ = z.loc[label_.index]
        mean_ = mean.loc[label_.index]
        logvar_ = logvar.loc[label_.index]
        dist_mat = distance_matrix(z_, mean_, logvar_)
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ps + '_distance_matrix.csv'))
        )
        dist_mat = distance_matrix(z_, mean_, logvar_, metric='NPD')
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ps + '_NPD_distance_matrix.csv'))
        )
        dist_mat = distance_matrix(z_, mean_, logvar_, metric='Euclidean')
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ps + '_Euclidean_distance_matrix.csv'))
        )
        dist_mat = distance_matrix(z_, mean_, logvar_, metric='Cosine')
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ps + '_Cosine_distance_matrix.csv'))
        )
        dist_mat = distance_matrix(z_, mean_, logvar_, metric='Pearson')
        dist_mat.to_csv(
            os.path.join('./data/output/distance_metrics',
                         (ps + '_Pearson_distance_matrix.csv'))
        )
