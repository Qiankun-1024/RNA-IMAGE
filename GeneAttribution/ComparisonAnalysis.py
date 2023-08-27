import os
import sys
import pandas as pd
import numpy as np

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(path)
from utils import load_model, gene2img
from .comparisonIG import comparision_attributions, ImportantGene


def group_data(labels, condition):
    grouped = labels.groupby(condition)
    group_name = labels[condition].unique()
    return grouped, group_name


def cross_group_comparison(model, images, xlabel, ylabel,
                           threshold, gene_list,
                           type='difference'):
    gene_attrs = pd.DataFrame()
    cofreq = pd.DataFrame(
        np.zeros((len(gene_list), len(gene_list))),
        dtype='int')
    cofreq.index, cofreq.columns = gene_list, gene_list

    for i in range(len(xlabel)):
        for j in range(len(ylabel)):
            attrs = comparision_attributions(model = model,
                                             X = images[xlabel.iloc[i]],
                                             Y = images[ylabel.iloc[j]],
                                             m_steps=50,
                                             output='1d')
            gene_attrs = pd.concat([gene_attrs, attrs], axis=1)
            
            ig = attrs_filter(attrs, threshold, type)
            cofreq.loc[ig.index, ig.index] += 1
    cofreq = cofreq.loc[~(cofreq==0).all(axis=1), ~(cofreq==0).all(axis=0)]
    return gene_attrs.T, cofreq


def intra_group_comparison(model, images, label,
                           threshold, gene_list,
                           type='marker'):
    gene_attrs = pd.DataFrame()
    cofreq = pd.DataFrame(
        np.zeros((len(gene_list), len(gene_list))),
        dtype='int')
    cofreq.index, cofreq.columns = gene_list, gene_list
    
    for i in range(len(label)-1):
        for j in range(i+1, len(label)):
            attrs = comparision_attributions(model = model,
                                             X = images[label.iloc[i]],
                                             Y = images[label.iloc[j]],
                                             m_steps=50,
                                             output='1d')
            gene_attrs = pd.concat([gene_attrs, attrs], axis=1)
            
            ig = attrs_filter(attrs, threshold, type)
            cofreq.loc[ig.index, ig.index] += 1
    
    cofreq = cofreq.loc[~(cofreq==0).all(axis=1), ~(cofreq==0).all(axis=0)]
    return gene_attrs.T, cofreq



def attrs_filter(attrs, threshold, type):
    if type == 'difference':
        attrs = attrs[attrs >= threshold]
        attrs.dropna(axis=0, inplace=True, how='all')
    if type == 'marker':
        attrs = attrs[attrs <= -threshold]
        attrs.dropna(axis=0, inplace=True, how='all')
    return attrs
    

def occurences_matirx(attrs, threshold, type):
    if type == 'difference':
        occurences = attrs.applymap(lambda x: 1 if x >= threshold else 0)
    if type == 'marker':
        occurences = attrs.applymap(lambda x: 1 if x <= -threshold else 0)
    return occurences


def merge_attrs_freq(attrs, occurrences):
    count = occurrences.sum()
    mean = attrs.mean()
    attrs = np.multiply(attrs, np.asarray(occurrences))
    important_mean = attrs.sum()/count
    freq = count/attrs.shape[0]
    
    result = pd.concat([mean, important_mean, freq], axis=1)
    result.fillna(0, inplace=True)
    result.iloc[:,2] = result.iloc[:,2].astype('int')
    result.columns = ['attrs', 'important_attrs', 'freq']
    return result
    

class ComparisonAnalysis():
    def __init__(self, datas, labels, dtype='TPM'):
        self.datas = datas
        self.labels = labels
        self.labels.loc[:,'idx'] = range(len(labels))
        self.model = load_model(only_encoder=True)
        self.images = gene2img(datas, dtype=dtype)
              
    def find_difference(self, condition, threshold=0.05, type='difference'):
        grouped, group_name = group_data(self.labels, condition)
        
        self._diff_attrs_ = {}
        self.diff_occurences = {}
        self.diff_cofreq = {}
        result = pd.DataFrame()
        for i in range(len(group_name)-1):
            for j in range(i+1,len(group_name)):
                group_X = grouped.get_group(group_name[i])
                group_Y = grouped.get_group(group_name[j])
                
                gene_attrs, cofreq, = cross_group_comparison(
                    self.model, self.images, group_X['idx'], group_Y['idx'], 
                    threshold, self.datas.columns, type
                    )
                id_x = [m for m in group_X.index for n in range(len(group_Y))]
                id_y = list(group_Y.index) * len(group_X)
                gene_attrs.index = [id_x, id_y]
                
                key = group_name[i] + '_vs_' + group_name[j]
                self._diff_attrs_[key] = gene_attrs
                
                occurences = occurences_matirx(gene_attrs, threshold, type)
                tmp_res = merge_attrs_freq(gene_attrs, occurences)
                tmp_res['group'] = key
                
                ig = attrs_filter(tmp_res[['attrs']], threshold, type)
                tmp_res.loc[ig.index, 'important'] = 1
                tmp_res.fillna(0, inplace=True)
                
                result = pd.concat([result, tmp_res], axis = 0)
                
                self.diff_occurences[key] = occurences
                self.diff_cofreq[key] = cofreq
                
        result.reset_index(inplace=True)
        result.columns = ['gene', 'attrs', 'important_attrs', 'freq', 'group', 'important']
        
        return result
        
    def find_marker(self, condition, threshold=0.05, type='marker'):
        grouped, group_name = group_data(self.labels, condition)
        
        self._marker_attrs_ = {}
        self.marker_occurences = {}
        self.marker_cofreq = {}
        result = pd.DataFrame()
        for i in range(len(group_name)):
            group = grouped.get_group(group_name[i])
                
            gene_attrs, cofreq, = intra_group_comparison(
                self.model, self.images, group['idx'],
                threshold, self.datas.columns, type
                )
            
            id_x, id_y = [], []
            for m in range(len(group)-1):
                for n in range(m+1, len(group)):
                    id_x.append(group.index[m])
                    id_y.append(group.index[n])
            gene_attrs.index = [id_x, id_y]
            
            key = group_name[i]
            self._marker_attrs_[key] = gene_attrs  
            
            occurences = occurences_matirx(gene_attrs, threshold, type)
            tmp_res = merge_attrs_freq(gene_attrs, occurences)
            tmp_res['group'] = key
            
            ig = attrs_filter(tmp_res[['attrs']], threshold, type)
            tmp_res.loc[ig.index, 'important'] = 1
            tmp_res.fillna(0, inplace=True)
            
            result = pd.concat([result, tmp_res], axis = 0)
            
            self.marker_occurences[key] = occurences
            self.marker_cofreq[key] = cofreq
            
        result.reset_index(inplace=True)
        result.columns = ['gene', 'attrs', 'important_attrs', 'freq', 'group', 'important']

        return result