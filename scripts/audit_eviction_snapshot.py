from pathlib import Path
import json
from lab.compare import load_matrix
from lab.audit import load_artifacts,verify_records
from lab.trace_audit import inspect_trace
from lab.activation import assess
p=Path('evidence/eviction-20261009a');base=p/'extracted/kv-pressure-run/kv-pressure-lab';root=base/'results/eviction-20261009a-eviction';work=base/'workloads/eviction-20261009a'
runs,missing=load_matrix(root,allow_partial=True,workload=work)
out={'missing_arms':missing,'complete_matrix':False,'arms':{}}
for arm in sorted(root.iterdir()):
 if not arm.is_dir():continue
 status=json.loads((arm/'status.json').read_text());policy=status['arm']['policy'];s,m=next((s,m) for s,m in runs if s['labels']['policy']==policy)
 a={'summary':s,'activation':assess((arm/'server.log').read_text(),policy),'status':status,'collection_integrity_verified':True}
 try:
  t=inspect_trace(arm,work);a['trace']=t;a['recomputed_positions']=sum(r['recompute_tokens'] for r in t['requests'].values());a['max_preemptions_per_request']=max(r['preemptions'] for r in t['requests'].values())
 except ValueError as e:a['trace_audit_error']=str(e)
 out['arms'][policy]=a
 print(policy,'throughput',s['completed_output_tokens_per_drain_inclusive_s'],'trace',a.get('trace_audit_error',a.get('recomputed_positions')))
 for name,x in s['phases'].items():print(name,{k:x[k] for k in ['arrivals','completed','ttft_p95_s','request_max_event_gap_p50_s','slo_passing_requests','completed_after_arrival_phase','waiting_queue_slope_per_s','quality_flags']})
a,b=[m['provenance']['server'] for _,m in runs]
for key in ['instance_id','image_id','model_revision','async_scheduling']:assert a[key]==b[key]
(p/'independent-partial-audit.json').write_text(json.dumps(out,indent=2));print('missing',missing)
