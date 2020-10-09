import pandas as pd
import numpy as np

absorbing_states = 2

mat_df = pd.read_csv("transition-matrix.csv", index_col=0, header = None)
mat_df.columns = mat_df.index
transition_mat = np.array(mat)

Q = transition_mat[0:(len(mat_df)-2),0:(len(mat_df)-2)]
R = transition_mat[0:(len(mat_df)-2),(len(mat_df)-2):len(mat_df)]
I = np.identity((len(mat_df)-2))

absorbing_probs = np.matmul(np.linalg.inv(I-Q),R)

