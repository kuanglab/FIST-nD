import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sources = 5
n = 1000
np.random.seed(0)

x = np.random.randn(n)
y = np.random.randn(n)
z = np.repeat([1,2,3,4],n//4)
spots = [f"{x}x{y}x{z}" for x, y, z in zip(x,y,z)]

df = pd.DataFrame({'spot':spots})

for gene in range(10):
    sources_x = np.random.uniform(-5, 5, sources)
    sources_y = np.random.uniform(-5, 5, sources)
    sources_z = np.random.uniform(0, 5, sources)
    strengths = np.random.exponential(size=sources)
    expressions = np.zeros(n)
    for i in range(sources):
        distances = (x-sources_x[i])**2 + (y-sources_z[i])**2 + (z-sources_z[i])**2
        expressions += np.round(1e1*np.exp(-distances/5))
    df[chr(gene+65)] = expressions

# we get pairwise correlations to simulate a PPI
expressions = df.to_numpy()[:, 1:].astype(int)
correlations = np.corrcoef(expressions.T)
PPI_entries = np.where(correlations>0.8)
entry_a = [chr(i+65) for i in PPI_entries[0]]
entry_b = [chr(i+65) for i in PPI_entries[1]]
PPI = pd.DataFrame({"A": entry_a, "B": entry_b})
PPI = PPI[PPI['A']!=PPI['B']]
PPI.columns = ["Official Symbol Interactor A", "Official Symbol Interactor B"]
PPI.to_csv('/project/compbioRAID1/Thomas/FIST-Python/example/PPI.txt', index=False, sep="\t")


# now we remove 70% of values for simulating dropout
expressions = df.to_numpy()[:, 1:]
indices = np.array(np.where(expressions>0)).T
zero_out = np.random.uniform(size=len(indices))<0.7
expressions[tuple(indices[zero_out, :].T)] = 0

df_with_dropout = pd.DataFrame(expressions.astype(int))
df_with_dropout.insert(0, 'spot', spots)
df_with_dropout.columns = df.columns
df_with_dropout.to_csv('/project/compbioRAID1/Thomas/FIST-Python/example/example_data.csv', index=False)
    