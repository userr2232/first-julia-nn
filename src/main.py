import pyarrow as pa
import numpy as np
import pyarrow.parquet as pq


# Write a Parquet file
arr = pa.array(np.arange(100))
# print(f"{arr[0]} .. {arr[-1]}")
table = pa.Table.from_arrays([arr], names=["col1"])
pq.write_table(table, "processed_data/example.parquet", compression=None)


# Reading a Parquet file
table = pq.read_table("processed_data/example.parquet")
# print(table)


# Reading a subset of Parquet data
table = pq.read_table("processed_data/example.parquet",
                        columns=["col1"],
                        filters=[
                            ("col1", ">", 5),
                            ("col1", "<", 10),
                        ])
# print(table)


# Saving Arrow Arrays to disk
arr = pa.array(np.arange(100))
# print(f"{arr[0]} .. {arr[-1]}")
schema = pa.schema([
    pa.field('nums', arr.type)
])

with pa.OSFile('processed_data/arraydata.arrow', 'wb') as sink:
    with pa.ipc.new_file(sink, schema=schema) as writer:
        batch = pa.record_batch([arr], schema=schema)
        writer.write(batch)


# Memory Mapping Arrow Arrays from disk
with pa.memory_map('processed_data/arraydata.arrow', 'r') as source:
    loaded_arrays = pa.ipc.open_file(source).read_all()

arr = loaded_arrays[0]
print(f"{arr[0]} .. {arr[-1]}")