import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import sys

def read_file(fname):
    if fname.endswith("parquet"):
        return pd.read_parquet(fname)
    else:
        return pd.read_csv(fname).set_index("spot")

before_df = read_file(sys.argv[1])
after_df = read_file(sys.argv[2])
gene = sys.argv[3]

if not(gene in before_df.columns and gene in after_df.columns):
    raise ValueError(f"Could not find gene {gene} in datasets.")

coords1 = before_df[before_df.columns[0]].to_numpy()
coords1 = np.array([c.split("x") for c in coords1]).astype(float)
ratio1 = np.ptp(coords1, axis=0)
coords2 = after_df[after_df.columns[0]].to_numpy()
coords2 = np.array([c.split("x") for c in coords2]).astype(float)
ratio2 = np.ptp(coords2, axis=0)

new_gene = 1 * (coords1[:, 0] > 0) * (coords1[:, 1] > 0)
print(new_gene)

fig = plt.figure()
ax1 = fig.add_subplot(121, projection='3d')
ax2 = fig.add_subplot(122, projection='3d')
ax1.scatter(*coords1.T ,c=np.log(new_gene+1), s=2.5)
ax1.set_box_aspect(ratio1)
ax2.scatter(*coords2.T ,c=np.log(new_gene+1), s=2.5)
ax2.set_box_aspect(ratio2)
print(coords1)
print(coords2)

plt.show()