import pandas as pd

from sparse_mmap import SparseCSR
from scipy.sparse import load_npz

HOME = 'data/'

df = pd.read_pickle(HOME + 'meta.pkl')
graph_df = pd.read_pickle(HOME + 'n2v_meta.pkl')

g = SparseCSR(
    HOME + 'g_row.npy',
    HOME + 'g_cols.npy',
    HOME + 'g_data.npy'
)
