"""Start the uploaded benchmark once; rerunning resumes completed cells."""
from pathlib import Path
import os,subprocess
p=Path('/content/exp121/scripts'); p.mkdir(parents=True,exist_ok=True)
script=p/'gpu_benchmark.py'; pidfile=p.parent/'benchmark.pid'
if not script.exists():
    print('Created directories; upload benchmark next')
else:
    active=False
    if pidfile.exists():
        pid=int(pidfile.read_text())
        cmd=Path(f'/proc/{pid}/cmdline')
        active=cmd.exists() and 'gpu_benchmark.py' in cmd.read_text()
    if active:
        print('Benchmark already running, PID',pid)
    else:
        log=(p.parent/'benchmark.log').open('a')
        proc=subprocess.Popen(['python','-u',str(script)],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        pidfile.write_text(str(proc.pid)); print('Benchmark PID',proc.pid)
