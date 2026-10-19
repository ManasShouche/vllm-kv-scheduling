"""Interpret scheduler observations without confusing enabled with pressure exercised."""
import re


def assess(log, policy):
    enabled = 'KVLAB_PPAS_INIT enabled=1' in log
    disabled = 'KVLAB_PPAS_INIT enabled=0' in log
    step = 'KVLAB_PPAS_STEP enabled=1' in log
    guarded = 'KVLAB_VARIANT name=decode_guard' in log
    threshold, cap = (1, 512) if policy == 'decode_guard' else (2, 2048)
    pressure = any(int(p) >= threshold and int(d) > 0 and int(b) == cap
                   for p, d, b in re.findall(r'KVLAB_PPAS_PRESSURE active_prefills=(\d+) decodes=(\d+) budget=(\d+)', log))
    adaptive = policy in ('ppas', 'decode_guard')
    valid = (enabled and step and not disabled) if adaptive else (disabled and not enabled and not step and not pressure)
    valid = valid and (guarded == (policy == 'decode_guard'))
    headroom = policy in ('reserve_static', 'reserve_feedback')
    markers = re.findall(r'KVLAB_HEADROOM event=(\w+) mode=(\w+)', log)
    if headroom:
        valid = valid and ('init', policy) in markers and ('step', policy) in markers
        valid = valid and all(mode == policy for _, mode in markers)
        pressure = ('defer', policy) in markers
    else:
        valid = valid and not markers
    eviction = policy in ('evict_cost', 'evict_fair')
    events = re.findall(r'KVLAB_EVICTION event=(\w+) mode=(\w+)', log)
    if eviction:
        valid = valid and ('init', policy) in events and all(mode == policy for _, mode in events)
        pressure = ('changed', policy) in events
    else:
        valid = valid and not events
    return {'policy': policy, 'activation_verified': valid,
            'pressure_branch_observed': pressure,
            'comparison_ready': valid and (not (adaptive or headroom or eviction) or pressure)}
