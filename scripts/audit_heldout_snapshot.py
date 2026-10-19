"""Independent frozen, per-seed held-out audit; no policy changes."""
from pathlib import Path
import json, hashlib
from lab.compare import load_matrix
from lab.trace_audit import inspect_trace
from lab.activation import assess
from lab.audit import load_artifacts
from lab.analyze import request_stats, percentile
p=Path('evidence/heldout-20261014a'); base=p/'extracted/kv-pressure-run/kv-pressure-lab'
root=base/'results/heldout-20261014a-heldout'; work=base/'workloads/heldout-20261014a'
lock=json.loads(Path('configs/cost-heldout-lock.json').read_text())
for f,h in lock['policy_files'].items(): assert hashlib.sha256((base/f).read_bytes()).hexdigest()==h
assert hashlib.sha256((base/'configs/cost-heldout.json').read_bytes()).hexdigest()==lock['config_sha256']
runs,missing=load_matrix(root,workload=work); assert not missing and len(runs)==6
out={}; rows_by={}
for arm in sorted(x for x in root.iterdir() if x.is_dir()):
 status=json.loads((arm/'status.json').read_text()); assert status['state']=='complete'
 a=status['arm']; key=f"{a['seed']}-{a['policy']}"
 assert {k:a[k] for k in ('policy','capacity','seed')}==lock['order'][len(out)]
 trace=inspect_trace(arm,work,allow_censored=True); assert trace['ready']
 activation=assess((arm/'server.log').read_text(),a['policy']); assert activation['activation_verified']
 summary,meta=next((s,m) for s,m in runs if s['labels']['policy']==a['policy'] and s['labels']['seed']==a['seed'])
 _,rows,_=load_artifacts(arm/'data'); rows_by[key]={r['request_id']:r for r in rows if not r['warmup']}
 out[key]={'summary':summary,'trace':trace,'activation':activation,'server':meta['provenance']['server']}
for k in ['image_id','instance_id','model_revision','async_scheduling']: assert len({v['server'][k] for v in out.values()})==1
(p/'independent-audits.json').write_text(json.dumps(out,indent=2))
gates={}; fairness={}
for seed in [201,202,203]:
 bk=f'{seed}-fixed_512'; ck=f'{seed}-evict_cost'; b=out[bk]; c=out[ck]
 bt=sum(r['recompute_tokens'] for r in b['trace']['requests'].values()); ct=sum(r['recompute_tokens'] for r in c['trace']['requests'].values())
 g={'baseline_recompute':bt,'candidate_recompute':ct,'recompute_ratio':ct/bt if bt else None,'mechanism_informative':bool(bt),'changed_victim':c['activation']['comparison_ready'],'throughput_ratio':c['summary']['completed_output_tokens_per_drain_inclusive_s']/b['summary']['completed_output_tokens_per_drain_inclusive_s'],'phases':{}}
 for phase,x in c['summary']['phases'].items():
  z=b['summary']['phases'][phase]; g['phases'][phase]={'ttft_p95_ratio':x['ttft_p95_s']/z['ttft_p95_s'],'gap_ratio':x['request_max_event_gap_p50_s']/z['request_max_event_gap_p50_s'],'slo_noninferior':x['slo_attainment_all_arrivals']>=z['slo_attainment_all_arrivals'],'no_extra_failures':x['arrivals']-x['completed']<=z['arrivals']-z['completed']}
 g['passes']=bool(bt and ct/bt<=.5 and g['changed_victim'] and g['throughput_ratio']>=.95 and all(v['ttft_p95_ratio']<=1.1 and v['gap_ratio']<=1.1 and v['slo_noninferior'] and v['no_extra_failures'] for v in g['phases'].values())); gates[seed]=g
 assert rows_by[bk].keys()==rows_by[ck].keys()
 pairs=[]
 for rid,r in rows_by[bk].items():
  q=rows_by[ck][rid]; assert (r['phase'],r['prompt_tokens'],r['output_tokens'])==(q['phase'],q['prompt_tokens'],q['output_tokens'])
  ok=r['status']==q['status']=='ok'; pairs.append({'request_id':rid,'phase':r['phase'],'prompt_tokens':r['prompt_tokens'],'censored':not ok,'ttft_delta_s':request_stats(q)['ttft_dispatch_s']-request_stats(r)['ttft_dispatch_s'] if ok else None,'completion_delta_s':(q['finished_s']-q['arrival_s'])-(r['finished_s']-r['arrival_s']) if ok else None})
 cohorts={}
 for name,cohort in [('all',pairs)]+[(f'{phase}-{length}',[r for r in pairs if r['phase']==phase and r['prompt_tokens']==length]) for phase in c['summary']['phases'] for length in [2048,8192,12288]]:
  v={'requests':len(cohort),'censored':sum(r['censored'] for r in cohort)}
  for metric in ['ttft_delta_s','completion_delta_s']:
   vals=[r[metric] for r in cohort if r[metric] is not None]; v[metric]={'median':percentile(vals,50),'p95':percentile(vals,95),'max':max(vals) if vals else None}
  cohorts[name]=v
 fairness[seed]={'pairs':pairs,'cohorts':cohorts,'preemptions':{label:{'max':max(r['preemptions'] for r in v['trace']['requests'].values()),'requests_with_two_or_more':sum(r['preemptions']>=2 for r in v['trace']['requests'].values())} for label,v in [('baseline',b),('candidate',c)]}}
(p/'independent-gates.json').write_text(json.dumps(gates,indent=2)); (p/'independent-fairness.json').write_text(json.dumps(fairness,indent=2)); print(json.dumps(gates,indent=2))
