import os
import time
import pandas as pd
import numpy as np
from metrics import *
from sklearn import metrics

class top_k_accuracy():
    def __init__(self, z, mean, logvar, label, k=1000):
        self.z, self.mean, self.logvar, self.label = z, mean, logvar, label
        self.k = k
        self.n = 100
        self.sample_index = label.sample(self.n, random_state=42).index
      
    def result(self, metric='Euclidean_Distances'):
        start_time = time.time()
        top_k_acc = np.zeros(self.k)
        sample_acc = []
        for i in self.sample_index:
            distance = []
            for j in self.label.index:
                x ,y = self.z.loc[i].values, self.z.loc[j].values
                x_mean, x_logvar = self.mean.loc[i].values, self.logvar.loc[i].values
                y_mean, y_logvar = self.mean.loc[j].values, self.logvar.loc[j].values
                if metric == 'Euclidean_Distances':
                    distance.append(Euclidean_Distances(x, y))
                if metric == 'Cosine_Distances':
                    distance.append(Cosine_Distances(x, y))
                if metric == 'Pearson_Distances':
                    distance.append(Pearson_Distances(x, y))
                if metric == 'KL_Divergence':
                    distance.append(KL_Divergence(x_mean, x_logvar,
                                                  y_mean, y_logvar))
                if metric == 'Wasserstein_Distance':
                    distance.append(Wasserstein_Distance(x_mean, x_logvar, 
                                                         y_mean, y_logvar))
                if metric == 'npd':
                    eps_x = np.random.normal(size=(50,len(x_mean)))
                    x_posterior = eps_x * np.exp(x_logvar * .5) + x_mean
                    eps_y = np.random.normal(size=(50,len(y_mean)))
                    y_posterior = eps_y * np.exp(y_logvar * .5) + y_mean
                    distance.append(npd(x, y, x_posterior, y_posterior))

            sort_distance = pd.DataFrame({metric:distance}, 
                                         index=self.label.index)
            sort_distance = sort_distance.sort_values(by=metric, ascending=True).head(self.k + 1)
            sort_label = self.label.loc[sort_distance.index]
            true_label = sort_label.drop(sort_label.index[0])
            label_counts = sort_label.drop(sort_label.index[0])
            for c in sort_label.columns:
                true_label[c] = true_label[c].apply(
                    lambda x : 1 if (x == self.label.loc[i,c]) & (x != 'None') else 0)
                label_counts[c] = label_counts[c].apply(
                    lambda x : 1 if (x != 'None') & (self.label.loc[i,c] != 'None') else 0)
            true_label = true_label.values
            label_counts = label_counts.values
            
            top_k_acc += np.sum(true_label, axis=1)/np.sum((label_counts + 1e-6), axis=1)
            sample_acc.append(np.sum(true_label, axis=0)/np.sum((label_counts + 1e-6), axis=0))
        top_k_acc = np.array(top_k_acc)
        top_k_acc = top_k_acc/ self.n
        sample_acc = pd.DataFrame(sample_acc, index=self.sample_index, columns=sort_label.columns)
        end_time = time.time()
        elapse_time = end_time - start_time

        return top_k_acc, elapse_time
    

    
if __name__ == '__main__':
    
    label = pd.read_csv('./data/TCGA_GTEx/y_data.tsv', index_col=0, sep='\t')
    label = label[label['cancer_type']!='Normal']
    label = label.groupby('cancer_type').sample(n=50)
    z = pd.read_csv('./data/output/zc.tsv', index_col=0, sep='\t')
    mean = pd.read_csv('./data/output/mean.tsv', index_col=0, sep='\t')
    logvar = pd.read_csv('./data/output/logvar.tsv', index_col=0, sep='\t')
    
    top_k_acc = top_k_accuracy(z, mean, logvar,
                                label[['cancer_type']])
    
    Euclidean_acc, Euclidean_time = top_k_acc.result(metric='Euclidean_Distances')
    Cos_acc, Cos_time = top_k_acc.result(metric='Cosine_Distances')
    Pearson_acc, Pearson_time = top_k_acc.result(metric='Pearson_Distances')
    Wasserstein_acc, Wasserstein_time = top_k_acc.result(metric='Wasserstein_Distance')
    npd_acc, npd_time = top_k_acc.result(metric='npd')
    
    acc = pd.DataFrame({'Euclidean_Distances':Euclidean_acc,
                            'Cosine_Distances':Cos_acc,
                            'Pearson_Distances':Pearson_acc,
                            'Wasserstein_Distance':Wasserstein_acc,
                            'npd':npd_acc}).T
    acc['time'] = [Euclidean_time, Cos_time, Pearson_time, Wasserstein_time, npd_time]
    acc.to_csv('./data/output/distance_metrics/top_k_ct_acc.tsv', sep='\t')       
        
    label = pd.read_csv('./data/TCGA_GTEx/y_data.tsv', index_col=0, sep='\t')
    label = label.groupby('primary_site').sample(n=50)
    z = pd.read_csv('./data/output/zc.tsv', index_col=0, sep='\t')
    mean = pd.read_csv('./data/output/mean.tsv', index_col=0, sep='\t')
    logvar = pd.read_csv('./data/output/logvar.tsv', index_col=0, sep='\t')
    
    top_k_acc = top_k_accuracy(z, mean, logvar,
                                label[['primary_site']])
    
    Euclidean_acc, Euclidean_time = top_k_acc.result(metric='Euclidean_Distances')
    Cos_acc, Cos_time = top_k_acc.result(metric='Cosine_Distances')
    Pearson_acc, Pearson_time = top_k_acc.result(metric='Pearson_Distances')
    Wasserstein_acc, Wasserstein_time = top_k_acc.result(metric='Wasserstein_Distance')
    npd_acc, npd_time = top_k_acc.result(metric='npd')
    
    acc = pd.DataFrame({'Euclidean_Distances':Euclidean_acc,
                            'Cosine_Distances':Cos_acc,
                            'Pearson_Distances':Pearson_acc,
                            'Wasserstein_Distance':Wasserstein_acc,
                            'npd':npd_acc}).T
    acc['time'] = [Euclidean_time, Cos_time, Pearson_time, Wasserstein_time, npd_time]
    acc.to_csv('./data/output/distance_metrics/top_k_ps_acc.tsv', sep='\t')