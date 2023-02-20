import pandas as pd
import numpy as np

label = pd.read_csv('./data/Breast/breast_label.tsv',
                    sep='\t', index_col=0)
cluster_label = pd.read_csv('./data/output/clustering/coare_grained/cluster_label.tsv',
                            index_col = 0, sep ='\t')
meta = pd.read_csv('./data/TCGA_GTEx/meta_info.tsv',
                    sep= '\t', index_col = 0)
cluster_label.index = meta.loc[cluster_label.index, 'entity_submitter_id'] 
tcga_label =  label.loc[cluster_label.index]
tcga_label = pd.concat([tcga_label, cluster_label], axis=1)
tcga_label['coarse_cluster_id'] = tcga_label['raw_data_0.2_KMeans']
tcga_label.drop(columns=['raw_data_0.2_KMeans'], inplace=True)
def clus_id(i):
    if i == 0:
        return "Luminal A1"
    if i == 1:
        return "Basal"
    if i == 2:
        return "Luminal A/B Mixed"
    if i == 3:
        return "Luminal A2"
    if i == 4:
        return "Her2/Luminal"
tcga_label['coarse_cluster'] = tcga_label['coarse_cluster_id'].map(clus_id)

tcga_label['fine_cluster'] = 0
for i in np.unique(tcga_label['coarse_cluster_id']):
    cluster_name = clus_id(i)
    cluster = pd.read_csv('./data/output/clustering/fine_grained/cluster_{}.tsv'.format(i),
                          sep='\t', index_col=0)
    cluster.columns = ['fine_cluster']
    # cluster['fine_cluster'] = cluster_name + '_' + cluster['fine_cluster'].astype(str)
    tcga_label.loc[cluster.index, 'fine_cluster'] = cluster['fine_cluster'].astype(str)
    

#delete Luminal A2_0 and Basal_0
tcga_label = tcga_label[(tcga_label['fine_cluster']!='0') | (tcga_label['coarse_cluster']!='Luminal A2')]
tcga_label = tcga_label[(tcga_label['fine_cluster']!='0') | (tcga_label['coarse_cluster']!='Basal')]

# Renumber
new_tcga_label = pd.DataFrame()
for coarse in np.unique(tcga_label['coarse_cluster']):
    fine_label = tcga_label.loc[tcga_label['coarse_cluster']==coarse,]
    old_label_name = np.unique(fine_label['fine_cluster'])
    new_label_name = [coarse + '_' + str(i) for i in range(1,len(old_label_name)+1)]
    fine_label = fine_label.replace(old_label_name, new_label_name)
    new_tcga_label = pd.concat([new_tcga_label, fine_label], axis=0)


other_label = label[(label['batch']!='TCGA') & (label['cancer_type']=='BRCA')]

data = pd.read_csv('./data/Breast/breast_data.tsv',
                   sep='\t', index_col=0)
tcga_data = data.loc[new_tcga_label.index]
other_data = data.loc[other_label.index]

new_tcga_label.to_csv('BRCA_classifier/data/tcga_label.tsv', sep='\t')
tcga_data.to_csv('BRCA_classifier/data/tcga_data.tsv', sep='\t')
other_label.to_csv('BRCA_classifier/data/other_label.tsv', sep='\t')
other_data.to_csv('BRCA_classifier/data/other_data.tsv', sep='\t')
