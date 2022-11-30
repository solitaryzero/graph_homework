import gzip
import csv
import pandas as pd

title2abs_file = './dataset/ogbn_arxiv/mapping/titleabs.tsv'
df = pd.read_csv(title2abs_file, sep='\t', header=None)
print(df)

# title2abs_file = './dataset/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz'
# df = pd.read_csv(title2abs_file, compression='gzip')
# print(df)

# title2abs_file = './dataset/ogbn_arxiv/raw/node-feat.csv.gz'
# df = pd.read_csv(title2abs_file, compression='gzip', header=None)
# print(df)