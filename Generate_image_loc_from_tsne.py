import tensorflow as tf
import pandas as pd
import numpy as np
import math
import os

import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def embed_gene_with_tsne(x):
    ##embedding gene into 2d-coordinate with t-sne
    pca = PCA(n_components=100)
    X_pc = pca.fit_transform(x.T)
    gene_embedded = TSNE(n_components=2, init='pca',
                            random_state=0, perplexity=10).fit_transform(X_pc)
    
    ##drop outliers
    gene_embedded, gene_index =drop_outliers(gene_embedded, x.columns)

    ##use convex hull to find the minimum box covering all the points
    hull = ConvexHull(gene_embedded)
    vertex_index = hull.vertices.tolist() #Indices of points forming the vertices of the convex hull
    vertex = gene_embedded[vertex_index]

    # find minimun rectangle
    polygon = Polygon(vertex) 
    min_rect = polygon.minimum_rotated_rectangle
    rect_points = np.array(min_rect.exterior.coords)

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.subplot(121)
    plt.scatter(gene_embedded[:, 0], gene_embedded[:, 1], s=5)
    plt.plot(rect_points[:,0], rect_points[:,1], 'r')
    for i in range(len(rect_points)-1):
        plt.text(rect_points[i,0], rect_points[i,1],str(i+1),fontsize=20)

    #rotate rectangle to horizontal direction 
    p_miny = rect_points[np.argmin(rect_points[:,1])]
    p_maxx = rect_points[np.argmax(rect_points[:,0])]
    tan_theta = (p_maxx[1]-p_miny[1]) / (p_maxx[0]-p_miny[0])
    theta = np.arctan(tan_theta)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta], [sin_theta, cos_theta]])
    rotated_gene_loc = np.matmul(gene_embedded, rotation_matrix)
    rotated_rect_points = np.matmul(rect_points, rotation_matrix)
    
    plt.subplot(122)
    plt.scatter(rotated_gene_loc[:, 0], rotated_gene_loc[:, 1], s=5)
    plt.plot(rotated_rect_points[:,0], rotated_rect_points[:,1], 'r')
    for i in range(len(rotated_rect_points)-1):
        plt.text(rotated_rect_points[i,0], rotated_rect_points[i,1],str(i+1),fontsize=20)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    fig_path = os.path.join(current_dir, 'figure')
    if not os.path.exists(fig_path):
        os.mkdir(fig_path)
    plt.savefig(os.path.join(fig_path, 'rotate_rectangle.png'))

    ##find minimun distance between all points
    CPP = Closest_Pair_of_Points(rotated_gene_loc)
    d_min = CPP.divide(0, len(rotated_gene_loc)-1)
    
    ##transform coordinates to pixel points
    px_size = 256 - 1
    x_max = np.max(rotated_rect_points[:,0])
    x_min = np.min(rotated_rect_points[:,0])
    y_max = np.max(rotated_rect_points[:,1])
    y_min = np.min(rotated_rect_points[:,1])
    L = (x_max - x_min)
    W = (y_max - y_min)
    precision = math.sqrt(2)
    A = math.ceil(L * precision / d_min)
    B = math.ceil(W * precision / d_min)
    if np.max([A,B]) > px_size:
        precision = precision * px_size/max([A,B])
        A = math.ceil(L * precision / d_min)
        B = math.ceil(W * precision / d_min)
    if np.max([A,B]) < px_size:
        precision = precision * px_size/max([A,B])
        A = math.ceil(L * precision / d_min)
        B = math.ceil(W * precision / d_min)

    if A < B:
        A = expand_to_2power(A)
    elif B < A:
        B = expand_to_2power(B)

    print('image precision is {}, minimum distance of points is {}'.format(precision, d_min))
    x_pixel = np.round(1 + A * (rotated_gene_loc[:,0] - x_min) / (x_max - x_min))
    y_pixel = np.round(1 + B * (rotated_gene_loc[:,1] - y_min) / (y_max - y_min))

    ##Move overlapping genes to surrounding pixels
    pixel_loc = np.stack((x_pixel, y_pixel), axis=1)
    pixel_loc_df = pd.DataFrame(pixel_loc)
    pixel_loc_df.index = gene_index

    return pixel_loc_df


def drop_outliers(data, index):
    df = pd.DataFrame(data)
    df.index = index
    p = df.boxplot(return_type='dict')
    x = p['fliers'][0].get_ydata()
    y = p['fliers'][1].get_ydata()

    df = df[~(df.iloc[:,0].isin(x) & (df.iloc[:,1].isin(y)))]

    return df.to_numpy(), df.index.to_list()



def expand_to_2power(integer):
    for i in range(7,9):
        if integer < (2**i-1):
            integer = 2 ** i - 1
            return integer



def uniq_sort_insert_pixels(pixel_loc_df):
    '''move around all duplicate gene pixels'''

    pixel_loc_df = pixel_loc_df.astype('int')
    uniq_df = pixel_loc_df.drop_duplicates(keep='first')

    duplicate_gene = list(set(pixel_loc_df.index).difference(set(uniq_df.index)))
    duplicate_df = pixel_loc_df.loc[duplicate_gene]
    print('The number of duplicate pixel points is {}, total gene number is {}'.format(len(duplicate_df), len(pixel_loc_df)))
    uniq_df = uniq_df.copy() 
    uniq_df['coords'] = [[x,y] for x, y in zip(uniq_df.iloc[:,0],uniq_df.iloc[:,1])]

    x_p_min = np.min(pixel_loc_df.iloc[:,0])
    x_p_max = np.max(pixel_loc_df.iloc[:,0])
    y_p_min = np.min(pixel_loc_df.iloc[:,1])
    y_p_max = np.max(pixel_loc_df.iloc[:,1])

    total_move_distance = 0
    max_move_distance = 0 
    for id in duplicate_gene:
        x = duplicate_df.loc[id][0]
        y = duplicate_df.loc[id][1]
        new_x, new_y, move_distance = move_around(x, y, x_p_max, x_p_min, y_p_max, y_p_min, uniq_df['coords'].to_list())
        uniq_df.loc[id] = [new_x, new_y, [new_x, new_y]]
        total_move_distance += move_distance
        if move_distance > max_move_distance:
            max_move_distance = move_distance

    print('max move step is {}, mean move step is {}'.format(max_move_distance, (total_move_distance/len(duplicate_gene))))   
    
    uniq_df.drop('coords', axis=1, inplace=True)
    uniq_df.columns = ['x_coord', 'y_coord']
    all_pixels = np.array([[i, j] for i in range(x_p_min,x_p_max+1) for j in range(y_p_min,y_p_max+1)])
    all_pixels_df = pd.DataFrame({'x_coord':all_pixels[:,0], 'y_coord':all_pixels[:,1]})
    all_pixels_df = pd.concat([uniq_df, all_pixels_df], join='outer')
    sorted_pixels_df = all_pixels_df.drop_duplicates(keep = 'first').sort_values(by=['x_coord', 'y_coord'])
    
    return sorted_pixels_df



def move_around(x, y, x_p_max, x_p_min, y_p_max, y_p_min, uniq_gene_list):
    searched_points = [(x,y)]
    for n in range(1, np.max([(x_p_max-x_p_min),(y_p_max-y_p_min)])):
        adjacent_points = [(i,j) for i in range(x-n, x+n+1) for j in range(y-n, y+n+1)]
        unsearched_points = list(set(adjacent_points).difference(set(searched_points)))
        searched_points = adjacent_points
        for i,j in unsearched_points:
            if (i >= x_p_min) & (i <= x_p_max) & (j >= y_p_min) & (j <= y_p_max):
                if [i,j] not in uniq_gene_list:
                    move_distance = n
                    return i, j, move_distance
                else:
                    continue

 
 
class Closest_Pair_of_Points():
    '''
    calculate the minimum distance between all gene coordinates
    '''

    def __init__(self, points):
        self.p = np.sort(points, axis=0)

    def distance(self, p1, p2):
        return math.sqrt(pow(p1[0] - p2[0], 2) + pow(p1[1] - p2[1], 2))
 
    def divide(self, start, end):
        if start == end:
            return float('inf')
        if (start + 1) == end:
            return self.distance(self.p[start], self.p[end])
                    
        mid = math.floor((start + end) / 2)
        d1 = self.divide(start, mid)
        d2 = self.divide(mid + 1, end)
        if d1 >= d2:
            d = d2
        else:
            d = d1

        b = self.p[mid][0]
        bl = b - d
        br = b + d

        for i in range(mid, start, -1):
            for j in range(mid, end, 1):
                if ((self.p[i][0] >= bl) & (self.p[i][0] <= b) & (self.p[j][0] <= br) & (self.p[j][0] > b)):
                    if math.fabs(self.p[i][1] - self.p[j][1]) < d:
                        dmid = self.distance(self.p[i], self.p[j])
                        if dmid < d:
                            d = dmid
                else:
                    continue
        return d


def generate_2d_loc(features):
    '''
    main process to generate 2d pixel points for each gene
    '''
    
    gene_pixels_df = embed_gene_with_tsne(features)
    sorted_pixels = uniq_sort_insert_pixels(gene_pixels_df)

    return sorted_pixels


if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data')
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    x_train = pd.read_csv(os.path.join(data_path, 'TCGA_GTEx/x_train.tsv'), index_col=0, sep='\t')
    sorted_pixels = generate_2d_loc(x_train)
    sorted_pixels.to_csv(os.path.join(data_path, 'tsne_sorted_pixels_coords.csv'))




    




    
    


