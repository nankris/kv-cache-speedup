"""Verify raw output files, paired identity, complete matrix and memory formula."""
from pathlib import Path
import csv,hashlib,itertools,json
root=Path(__file__).resolve().parents[1]/'results/gpu'
assert (root/'0_6B/timings.csv').exists()
for d in [root/'0_6B',root/'1_7B']:
    if not (d/'timings.csv').exists(): continue
    rows=list(csv.DictReader((d/'timings.csv').open()))
    expected=set(itertools.product([64,512],[32,64,128,256,512],range(3),[True,False]))
    seen=set(); outputs={}
    for r in rows:
        p,n,rep=int(r['prompt_tokens']),int(r['output_tokens']),int(r['repeat']); cache=r['use_cache']=='True'
        key=(p,n,rep,cache); assert key not in seen; seen.add(key)
        ids=json.loads((d/f'output_p{p}_n{n}_r{rep}_c{int(cache)}.json').read_text())
        assert len(ids)==n==int(r['generated_tokens'])
        assert hashlib.sha256(json.dumps(ids).encode()).hexdigest()==r['output_sha256']
        assert float(r['seconds'])>0
        assert abs(float(r['tokens_per_second'])-n/float(r['seconds']))<1e-8
        if (p,n) in outputs: assert ids==outputs[p,n], f'Mismatch {key}'
        outputs[p,n]=ids
    assert seen==expected, expected-seen
    for m in json.loads((d/'memory.json').read_text()):
        assert m['formula_bytes']==m['tensor_bytes']==m['storage_bytes']
        assert m['live_bytes']-m['baseline_bytes']==m['delta_bytes']
    print(d.name,': 60 rows; all saved token IDs and hashes match; memory formula exact')
mechanism=json.loads((root/'mechanism_check.json').read_text())
assert [[c['input_shape'][1] for c in r['calls']] for r in mechanism]==[[64,1,1,1],[64,65,66,67]]
print('Mechanism: cache reuses prefix; no-cache recomputes growing prefix')
