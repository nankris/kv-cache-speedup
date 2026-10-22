"""Run the untimed mechanism check immediately after the primary benchmark."""
from pathlib import Path
import subprocess
root=Path('/content/kvcache')
code='''from pathlib import Path
import subprocess,time,json
root=Path('/content/kvcache')
while not (root/'results/gpu/primary_complete.json').exists(): time.sleep(1)
with (root/'diagnostic.log').open('w') as f:
    p=subprocess.run(['python','-u',str(root/'scripts/gpu_diagnostic.py')],stdout=f,stderr=subprocess.STDOUT)
(root/'results/gpu/diagnostic_status.json').write_text(json.dumps(dict(returncode=p.returncode)))
'''
p=subprocess.Popen(['python','-u','-c',code],start_new_session=True)
print('Final diagnostic watcher PID',p.pid)
