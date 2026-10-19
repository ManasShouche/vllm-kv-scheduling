"""CPU-only decision kernel for a proposed memory/deadline feasibility study.

Not wired into vLLM. Bounds must come from a separately validated profiler.
Empirical upper estimates are NOT hard real-time guarantees. This module neither
fits them nor treats historical client traces as counterfactual GPU execution.
"""
from dataclasses import dataclass
import math


def nonnegative(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def integer(value, name):
    nonnegative(value, name)
    if not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')


@dataclass(frozen=True)
class Option:
    prefill_tokens: int
    step_upper_ms: float
    # Remaining wall time to the chosen prefill's first output, including this
    # step, subsequent prefill steps and first decode; not merely step duration.
    first_response_upper_ms: float
    additional_kv_blocks: int

    def __post_init__(self):
        integer(self.prefill_tokens, 'prefill_tokens')
        integer(self.additional_kv_blocks, 'additional_kv_blocks')
        nonnegative(self.step_upper_ms, 'step_upper_ms')
        nonnegative(self.first_response_upper_ms, 'first_response_upper_ms')
        if self.first_response_upper_ms < self.step_upper_ms:
            raise ValueError('First-response bound must include the next step')


@dataclass(frozen=True)
class State:
    decode_slack_ms: float | None
    prefill_slack_ms: float
    free_kv_blocks: int
    # Reserved future growth of current decodes, including the next step.
    # No output-length oracle is assumed; the chosen horizon must be recorded.
    reserved_decode_blocks: int
    calibrated: bool

    def __post_init__(self):
        for name in ('decode_slack_ms', 'prefill_slack_ms'):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError(f'{name} must be finite; negative means overdue')
        if self.prefill_slack_ms is None:
            raise ValueError('Prefill deadline is required')
        integer(self.free_kv_blocks, 'free_kv_blocks')
        integer(self.reserved_decode_blocks, 'reserved_decode_blocks')
        if not isinstance(self.calibrated, bool):
            raise ValueError('calibrated must be a boolean')


def choose(options, state):
    """Enumerate measured actions; do not assume monotone GPU kernel timings.

If the joint set is empty, preserve a feasible decode-only step and explicitly
report the TTFT conflict. If that too is infeasible, hand back to an overload
policy. No request is silently dropped and no safety guarantee is fabricated.
"""
    if not options or len({o.prefill_tokens for o in options}) != len(options):
        raise ValueError('Options must be nonempty with distinct token counts')
    if not any(o.prefill_tokens == 0 for o in options):
        raise ValueError('A measured decode-only option is required')
    if not state.calibrated:
        return {'action': 'requires_calibration', 'prefill_tokens': None,
                'joint_model_feasible': False, 'rejected': []}
    feasible, decode_only, rejected = [], None, []
    for option in options:
        reasons = []
        if state.decode_slack_ms is not None and option.step_upper_ms > state.decode_slack_ms:
            reasons.append('decode_deadline')
        if option.additional_kv_blocks + state.reserved_decode_blocks > state.free_kv_blocks:
            reasons.append('kv_headroom')
        if option.prefill_tokens == 0 and not reasons:
            decode_only = option
        if option.first_response_upper_ms > state.prefill_slack_ms:
            reasons.append('prefill_deadline')
        if reasons:
            rejected.append({'prefill_tokens': option.prefill_tokens, 'constraints': reasons})
        elif option.prefill_tokens > 0:
            feasible.append(option)
    if feasible:
        selected = max(feasible, key=lambda o: o.prefill_tokens)
        return {'action': 'prefill', 'prefill_tokens': selected.prefill_tokens,
                'joint_model_feasible': True, 'rejected': rejected}
    return {'action': 'defer_prefill' if decode_only else 'overload',
            'prefill_tokens': 0 if decode_only else None,
            'joint_model_feasible': False, 'rejected': rejected}
