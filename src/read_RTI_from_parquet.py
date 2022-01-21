import base64
import pyarrow.parquet as pq
import numpy as np
import pyarrow as pa
from pathlib import Path
import matplotlib.pyplot as plt

path = Path('processed_data/parquet/RTIs.parquet')
table = pq.read_table(path, columns=['2002_Dec_29'])

df = table.to_pandas()
# arr = np.ascontiguousarray(df.values.flatten().reshape(720, 120).T)
arr = df.values.reshape(720, 120).T
fig = plt.figure(figsize=(60, 10))

ax1 = fig.add_subplot(121)
ax1.set_title('colorMap')
plt.imshow(arr)
ax1.set_aspect('equal')

# x, y = arr.shape

arr2 = np.nansum(np.nansum(arr.reshape(30, 4, 48, 15), 1), 2) != 0

ax2 = fig.add_subplot(122)
ax2.set_title('colorMap')
plt.imshow(arr2)
ax2.set_aspect(1/6)

plt.show()
