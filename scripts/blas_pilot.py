import ctypes,time,json
from pathlib import Path
import os,torch
lib=ctypes.CDLL(os.environ.get('OPENBLAS_LIB',os.path.join(os.path.dirname(torch.__file__),'lib','libopenblas.so.0')))
m,n,k=512,3072,1024
A=(ctypes.c_float*(m*k))();B=(ctypes.c_float*(k*n))();C=(ctypes.c_float*(m*n))()
rows=[]
for threads in [1,2]:
 lib.openblas_set_num_threads(threads)
 for repeat in range(3):
  t=time.perf_counter()
  lib.cblas_sgemm(101,111,111,m,n,k,ctypes.c_float(1),A,k,B,n,ctypes.c_float(0),C,n)
  rows.append(dict(threads=threads,repeat=repeat,seconds=time.perf_counter()-t))
  print(rows[-1],flush=True)
Path('results/blas_pilot.json').write_text(json.dumps(rows,indent=2))
