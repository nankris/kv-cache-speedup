"""Stop this benchmark after its primary matrix; optional larger model is not quick."""
from pathlib import Path
import subprocess
root=Path('/content/kvcache')
assert (root/'benchmark.pid').exists(), 'Launch benchmark first'
code='''from pathlib import Path
import os,signal,time,json
root=Path('/content/kvcache'); pid=int((root/'benchmark.pid').read_text())
while not (root/'results/gpu/0_6B/memory.json').exists():
    cmd=Path(f'/proc/{pid}/cmdline')
    if not cmd.exists() or 'gpu_benchmark.py' not in cmd.read_text(): raise SystemExit('Benchmark exited before primary completion')
    time.sleep(1)
cmd=Path(f'/proc/{pid}/cmdline')
if cmd.exists() and 'gpu_benchmark.py' in cmd.read_text(): os.kill(pid,signal.SIGTERM)
(root/'results/gpu/primary_complete.json').write_text(json.dumps(dict(primary_complete=True,optional_model='not run: FP32 primary timings imply larger-model full matrix is not quick',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),indent=2))
'''
p=subprocess.Popen(['python','-u','-c',code],stdout=(root/'watcher.log').open('w'),stderr=subprocess.STDOUT,start_new_session=True)
print('Primary completion watcher PID',p.pid)
