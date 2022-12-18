import pandas as pd
import matplotlib.pyplot as plt

from load_model import generate_data
from evaluate import data_visualization


def load_data():
    label = pd.read_csv('./data/all_label.tsv', index_col=0, sep='\t')
    data = pd.read_csv('./data/log2_x_data.tsv', index_col=0, sep='\t')
    label = label[label['primary_site'].isin(['Kidney'])
                  & (label['cancer_type']=='Normal')]
    print(label)
    data = data.loc[label.index]
    data = 2 ** data - 1
    return data, label

if __name__ == '__main__':
    data, label = load_data()
    z, bf_x = generate_data(data, label)
    data, label = data_visualization.data_filter(
        data, label, {'primary_site':['Kidney']}
        )
    z, label = data_visualization.data_filter(
        data, label, {'primary_site':['Kidney']}
        )
    bf_x, label = data_visualization.data_filter(
        data, label, {'primary_site':['Kidney']}
        )
    plt_classes = ['batch', 'cancer_type', 'tumor_stage'] 
    fig, _ = data_visualization.umap(data, label, plt_classes)
    plt.savefig('./figure/partial_visualization/x_Kidney_umap.png')
    plt.close()
    fig, _ = data_visualization.umap(z, label, plt_classes)
    plt.savefig('./figure/partial_visualization/z_Kidney_umap.png')
    plt.close()
    fig, _ = data_visualization.umap(bf_x, label, plt_classes)
    plt.savefig('./figure/partial_visualization/bfx_Kidney_umap.png')
    plt.close()