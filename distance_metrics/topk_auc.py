import os
import time
import pandas as pd
import numpy as np
from metrics import *
from sklearn import metrics


class top_k_roc():
    def __init__(self, mean, logvar, label, k=200):
        self.mean, self.logvar, self.label = mean, logvar, label
        
        start_time = time.time()
        eps = np.random.normal(size=mean.shape)
        self.z =  eps * np.exp(logvar * .5) + mean
        end_time = time.time()
        self.sample_time = end_time - start_time
        
        self.k = k
        self.n = 100
        self.sample_index = label.sample(self.n, random_state=42).index
      
    def result(self, metric='Euclidean_Distances'):
        start_time = time.time()
        pred_label = np.zeros(self.n * self.k)
        true_label = np.zeros((len(self.label.columns), self.n * self.k))
        count = 0
        top_roc_auc = pd.DataFrame()
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
            true = sort_label.drop(sort_label.index[0])
            pred = [i/self.k for i in range(len(true),0,-1)]

            for c in sort_label.columns:
                true[c] = true[c].apply(
                    lambda x : 1 if (x == self.label.loc[i,c]) & (x != 'None') else 0)
            true = true.values.T
            
            start = count * self.k
            end = start + self.k
            pred_label[start:end] = pred
            true_label[:,start:end] = true
            count += 1

        end_time = time.time()
        elapse_time = end_time - start_time
        
        if metric in ['Wasserstein_Distance', 'KL_Divergence'] :
            elapse_time = elapse_time
        else:
            elapse_time = elapse_time + self.sample_time
        
        for row in range(len(self.label.columns)):
            fpr, tpr, thresholds = metrics.roc_curve(
                true_label[row,:], pred_label, pos_label=1)
            auc = metrics.auc(fpr, tpr)
            df = pd.DataFrame({'fpr': fpr,
                               'tpr': tpr,
                               'thresholds': thresholds,
                               'auc':auc,
                               'label':self.label.columns[row]})
            top_roc_auc = top_roc_auc.append(df)
        top_roc_auc['time'] = elapse_time
        top_roc_auc['metric'] = metric

        return top_roc_auc
    


class top_k_auc():
    def __init__(self, mean, logvar, label, query_label, k=1000):
        self.mean, self.logvar, self.label, self.query_label = mean, logvar, label, query_label
        
        start_time = time.time()
        eps = np.random.normal(size=mean.shape)
        self.z =  eps * np.exp(logvar * .5) + mean
        end_time = time.time()
        self.sample_time = end_time - start_time
        
        self.k = k
        self.n = len(query_label)
        self.sample_index = query_label.index
      
    def result(self, metric):
        start_time = time.time()

        auc_score = np.zeros((len(self.label.columns), self.n))
        count = 0
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
                    eps_x = np.random.normal(size=(500000,len(x_mean)))
                    x_posterior = eps_x * np.exp(x_logvar * .5) + x_mean
                    eps_y = np.random.normal(size=(500000,len(y_mean)))
                    y_posterior = eps_y * np.exp(y_logvar * .5) + y_mean
                    distance.append(npd(x, y, x_posterior, y_posterior))

            sort_distance = pd.DataFrame({metric:distance}, 
                                         index=self.label.index)
            sort_distance = sort_distance.sort_values(by=metric, ascending=True).head(self.k)
            sort_label = self.label.loc[sort_distance.index]
            true = sort_label
            # pred = (1/(1 + sort_distance)).T.values.flatten()
            pred = np.exp(-sort_distance/20).T.values.flatten()

            # pred = [i/self.k for i in range(len(true),0,-1)]

            for c in sort_label.columns:
                true[c] = true[c].apply(
                    lambda x : 1 if (x == self.query_label.loc[i,c]) & (x != 'None') else 0)
            true = true.values.T
            
            for row in range(len(self.label.columns)):
                fpr, tpr, thresholds = metrics.roc_curve(
                    true[row,:], pred, pos_label=1)
                auc = metrics.auc(fpr, tpr)
                auc_score[row, count] = auc
            count += 1

        end_time = time.time()
        elapse_time = end_time - start_time
        
        if metric in ['Wasserstein_Distance', 'KL_Divergence'] :
            elapse_time = elapse_time
        else:
            elapse_time = elapse_time + self.sample_time
        
        topk_auc = pd.DataFrame(auc_score.T)
        topk_auc.columns = ['auc']
        topk_auc['time'] = elapse_time
        topk_auc['metric'] = metric
        print(topk_auc)

        return topk_auc


    
if __name__ == '__main__':

    
    # label = pd.read_csv('./data/TCGA_GTEx/y_data.tsv', index_col=0, sep='\t')
    
    # breast_label = pd.read_csv('./data/Breast/breast_label.tsv', index_col=0, sep='\t')
    # breast_label = breast_label[breast_label['batch']!='TCGA']
    # breast_label = breast_label[breast_label['cancer_type'] == 'BRCA']
    # breast_label = breast_label.sample(n=50, random_state=0)
    
    # mean = pd.read_csv('./data/output/mean.tsv', index_col=0, sep='\t')
    # logvar = pd.read_csv('./data/output/logvar.tsv', index_col=0, sep='\t')
    # breast_mean = pd.read_csv('./data/output/breast_mean.tsv', index_col=0, sep='\t')
    # breast_logvar = pd.read_csv('./data/output/breast_logvar.tsv', index_col=0, sep='\t')
    # breast_mean = breast_mean.loc[breast_label.index]
    # breast_logvar = breast_logvar.loc[breast_label.index] 
    # mean = pd.concat([mean, breast_mean])
    # logvar = pd.concat([logvar, breast_logvar])
    
    # topauc = top_k_auc(mean, logvar,
    #                         label[['cancer_type']], breast_label[['cancer_type']])
    
    # Euclidean_auc = topauc.result(metric='Euclidean_Distances')
    # Cos_auc = topauc.result(metric='Cosine_Distances')
    # Pearson_auc = topauc.result(metric='Pearson_Distances')
    # Wasserstein_auc = topauc.result(metric='Wasserstein_Distance')
    # npd_auc = topauc.result(metric='npd')
    
    # roc_auc = pd.concat([Euclidean_auc, Cos_auc, Pearson_auc,
    #                     Wasserstein_auc, npd_auc])
    # roc_auc.to_csv('./data/output/distance_metrics/top_k_breast_auc.tsv', sep='\t')    
    
    
    label = pd.read_csv('./data/TCGA_GTEx/y_data.tsv', index_col=0, sep='\t')
    KIRP_label = label[label['cancer_type']=="KIRP"].sample(n=10, random_state=42)
    
    mean = pd.read_csv('./data/output/mean.tsv', index_col=0, sep='\t')
    logvar = pd.read_csv('./data/output/logvar.tsv', index_col=0, sep='\t')
    KIRP_mean = mean.loc[KIRP_label.index]
    KIRP_logvar= logvar.loc[KIRP_label.index]
    KIRP_label.index = [i for i in range(10)]
    KIRP_mean.index = [i for i in range(10)]
    KIRP_logvar.index = [i for i in range(10)]

    mean = pd.concat([mean, KIRP_mean], axis=0)
    logvar = pd.concat([logvar, KIRP_logvar], axis=0)
    
    topauc = top_k_auc(mean, logvar,
                       label[['cancer_type']], KIRP_label[['cancer_type']],
                       k=50)
    
    Euclidean_auc = topauc.result(metric='Euclidean_Distances')
    Cos_auc = topauc.result(metric='Cosine_Distances')
    Pearson_auc = topauc.result(metric='Pearson_Distances')
    Wasserstein_auc = topauc.result(metric='Wasserstein_Distance')
    npd_auc = topauc.result(metric='npd')
    
    roc_auc = pd.concat([Euclidean_auc, Cos_auc, Pearson_auc,
                        Wasserstein_auc, npd_auc])
    roc_auc.to_csv('./data/output/distance_metrics/top_k_KIRP_auc.tsv', sep='\t')    
    

