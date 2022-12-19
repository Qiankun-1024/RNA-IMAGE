from collections import Counter
import pandas as pd
import numpy as np
import os
import re

clinical = pd.read_csv('/vol1/cuipeng_group/qiankun/GAN-VAE/data/clinical.tsv',
                       index_col=0, sep='\t')
print(clinical.tumor_grade.value_counts())