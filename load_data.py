import pandas as pd

from scipy.sparse import load_npz

HOME = 'data/'

g = load_npz(HOME + 'graph.npz')
df = pd.read_pickle(HOME + 'meta.pkl')
graph_df = pd.read_pickle(HOME + 'n2v_meta.pkl')
