"""Historical operation: stop only this experiment's BF16 process and archive it."""
from pathlib import Path
import os,signal,time
root=Path('/content/exp121'); env=root/'results/gpu/0_6B/environment.json'
import json
assert json.loads(env.read_text())['dtype']=='bfloat16', 'Refusing to archive a non-BF16 run'
pid=int((root/'benchmark.pid').read_text())
assert 'gpu_benchmark.py' in Path(f'/proc/{pid}/cmdline').read_text()
os.kill(pid,signal.SIGTERM); time.sleep(2)
(root/'results/gpu').rename(root/'results/gpu_bf16_exploratory')
print('Stopped own BF16 process; preserved completed rows')
