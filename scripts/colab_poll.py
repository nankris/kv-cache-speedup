from pathlib import Path
import subprocess,tarfile
root=Path('/content/exp121')
p=root/'benchmark.log'
print(p.read_text()[-1500:] if p.exists() else 'No log yet')
for p in (root/'results/gpu').glob('*/timings.csv'):
    print(str(p),len(p.read_text().splitlines())-1,'rows')
print(subprocess.getoutput('nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv'))
with tarfile.open('/content/exp121_checkpoint.tar.gz','w:gz') as t:
    for p in [root/'results',root/'benchmark.log',root/'diagnostic.log',root/'watcher.log']:
        if p.exists(): t.add(p,arcname=p.name)
