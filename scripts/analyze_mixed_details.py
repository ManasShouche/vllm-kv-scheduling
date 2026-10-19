from pathlib import Path
import json
from lab.audit import load_artifacts
from lab.analyze import request_stats,percentile
from lab.eviction import victim_index
from types import SimpleNamespace as O
p=Path('evidence/mixed-20261013a');root=p/'extracted/kv-pressure-run/kv-pressure-lab/results/mixed-20261013a-eviction';out={};table=[]
for arm in sorted(root.iterdir()):
 if not arm.is_dir():continue
 meta,rows,_=load_artifacts(arm/'data');policy=meta['labels']['policy'];dec=[];log=(arm/'server.log').read_text()
 for line in log.splitlines():
  if 'KVLAB_EVICTION_DECISION ' in line:
   d=json.loads(line.split('KVLAB_EVICTION_DECISION ',1)[1]);rs=[O(num_computed_tokens=r['computed'],num_preemptions=r['preemptions'],i=i) for i,r in enumerate(d['requests'])];assert victim_index(rs,d['cursor'],lambda r:d['requests'][r.i]['blocks'],policy=='evict_fair')==d['selected'];dec.append(d)
 cohorts=[]
 for phase in meta['config']['phases']:
  for length in [2048,8192,12288]:
   cohort=[r for r in rows if not r['warmup'] and r['phase']==phase['name'] and r['prompt_tokens']==length];stats=[request_stats(r) for r in cohort if r['status']=='ok'];passed=sum(x['ttft_dispatch_s']<=meta['config']['ttft_slo_s'] and x['max_event_gap_s'] is not None and x['max_event_gap_s']<=meta['config']['event_gap_slo_s'] for x in stats)
   x={'phase':phase['name'],'prompt_tokens':length,'arrivals':len(cohort),'completed':len(stats),'slo_passes':passed,'ttft_p95_s':percentile([x['ttft_dispatch_s'] for x in stats],95)};cohorts.append(x);table.append(f"| {policy} | {phase['name']} | {length} | {len(stats)}/{len(cohort)} | {passed}/{len(cohort)} | {x['ttft_p95_s']:.3f} |")
 out[policy]={'requests':len(rows),'decisions':dec,'changed':sum(d['selected']!=d['default'] for d in dec),'truncated':'event=diagnostics_truncated' in log,'by_length':cohorts}
(p/'independent-victim-and-length.json').write_text(json.dumps(out,indent=2));Path('/tmp/mixed-table.md').write_text('\n'.join(table));print({k:{x:v[x] for x in ['requests','changed','truncated']} for k,v in out.items()})
