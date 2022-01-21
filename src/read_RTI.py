import pandas as pd
import pyarrow as pa
from pyarrow import plasma
from pyarrow import ipc
import numpy as np

file_name = "data/Data_Julia/Julia_ESF_2002.txt"
df = pd.read_table(file_name, sep='\s+', \
                            na_values='missing', low_memory=True, \
                            dtype={'UT1_UNIX': np.int64, 'GDALT': float, 'SNL': float})
UTC5_offset = 5 * 60 * 60
df['datetime'] = pd.to_datetime(df['UT1_UNIX']-UTC5_offset, unit='s')
df.drop(columns=['UT1_UNIX'], inplace=True)
df.sort_values(by='datetime', inplace=True)
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)
table = pa.Table.from_pandas(df)
print(table.schema)
client = plasma.connect('/tmp/plasma')
random_num = np.random.bytes(20)
print("random_num", random_num)
object_id = plasma.ObjectID(random_num)
sink = pa.MockOutputStream()
with pa.RecordBatchStreamWriter(sink, table.schema) as stream_writer:
    stream_writer.write_table(table)
data_size = sink.size()
buf = client.create(object_id, data_size)
stream = pa.FixedSizeBufferWriter(buf)
with pa.RecordBatchStreamWriter(stream, table.schema) as stream_writer:
    stream_writer.write_table(table)
client.seal(object_id)
print("sealed object_id", object_id)
# for date in pd.date_range('1/1/2002', '31/12/2002'):
#     start = pd.Timestamp(date) + pd.Timedelta('19h')
#     end = pd.Timestamp(date) + pd.Timedelta('1d') + pd.Timedelta('7h')
#     l = len(df.loc[((df.datetime >=start)&(df.datetime<end)&(df.GDALT > 200)&(df.GDALT < 800))])
#     if l: print(start, end, l)
#     # print(start, end)






















# import pyarrow as pa
# import pyarrow.plasma as plasma
# from pyarrow import ipc
# import numpy as np

# plasma_client = plasma.connect('/tmp/plasma')

# # inputs: a list of object ids
# inputs = [20 * b'1']

# # Construct Object ID and perform a batch get
# object_ids = [plasma.ObjectID(inp) for inp in inputs]
# buffers = plasma_client.get_buffers(object_ids)

# # Read the tensor and convert to numpy array for each object
# arrs = []
# for buffer in buffers:
#     reader = pa.BufferReader(buffer)
#     t = ipc.read_tensor(reader)
#     arr = t.to_numpy()
#     arrs.append(arr)

# # arrs is now a list of numpy arrays
# assert np.all(arrs[0] == 2.0 * np.ones(1000, dtype="float32"))