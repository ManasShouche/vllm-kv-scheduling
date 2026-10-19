from pathlib import Path
import json
from lab.compare import load_matrix
from lab.trace_audit import inspect_trace
from lab.activation import assess
p=Path('evidence/mixed-20261013a');base=p/'extracted/kv-pressure-run/kv-pressure-lab';root=base/'results/mixed-20261013a-eviction';work=base/'workloads/mixed-20261013a'
runs,missing=load_matrix(root,workload=work);assert not missing
out={}
for arm in sorted(root.iterdir()):
 if not arm.is_dir():continue
 status=json.loads((arm/'status.json').read_text());assert status['state']=='complete'
 policy=status['arm']['policy'];trace=inspect_trace(arm,work)
 activation=assess((arm/'server.log').read_text(),policy);assert activation['activation_verified'];print('activation',policy,activation,'max_repeat',max(r['preemptions'] for r in trace['requests'].values()))
 summary,meta=next((a,b) for a,b in runs if a['labels']['policy']==policy)
 print(policy, list(arm.glob('*log')))
 out[policy]={'activation':activation,'summary':summary,'trace':trace,'server':meta['provenance']['server']}
for key in ['image_id','instance_id','model_revision','async_scheduling']:
 assert len({v['server'][key] for v in out.values()})==1,key
(p/'independent-audits.json').write_text(json.dumps(out,indent=2))
b=out['fixed_512'];bt=sum(r['recompute_tokens'] for r in b['trace']['requests'].values());gates={}
for name,v in out.items():
 s=v['summary']; rt=sum(r['recompute_tokens'] for r in v['trace']['requests'].values());print(name,'recompute',rt,'preempt',v['trace']['preemptions'],'throughput',s['completed_output_tokens_per_drain_inclusive_s'])
 g={'recomputed_positions':rt,'preemptions':v['trace']['preemptions'],'recompute_ratio':rt/bt,'throughput_ratio':s['completed_output_tokens_per_drain_inclusive_s']/b['summary']['completed_output_tokens_per_drain_inclusive_s'],'phases':{}}
 for phase,x in s['phases'].items():
  z=b['summary']['phases'][phase];g['phases'][phase]={'ttft_p95_ratio':x['ttft_p95_s']/z['ttft_p95_s'],'median_worst_gap_ratio':x['request_max_event_gap_p50_s']/z['request_max_event_gap_p50_s'],'slo_noninferior':x['slo_attainment_all_arrivals']>=z['slo_attainment_all_arrivals'],'no_extra_failures':x['arrivals']-x['completed']<=z['arrivals']-z['completed']}
  print(phase,{k:x[k] for k in ['arrivals','completed','slo_passing_requests','ttft_p95_s','request_max_event_gap_p50_s','quality_flags','waiting_queue_slope_per_s','completed_after_arrival_phase']})
 g['passes']=g['recompute_ratio']<=.5 and g['throughput_ratio']>=.95 and all(x['ttft_p95_ratio']<=1.10 and x['median_worst_gap_ratio']<=1.10 and x['slo_noninferior'] and x['no_extra_failures'] for x in g['phases'].values());gates[name]=g
gates['fixed_512']['passes'] = None
gates['fixed_512']['role'] = 'reference'
(p/'independent-gates.json').write_text(json.dumps(gates,indent=2));print(json.dumps(gates,indent=2))
