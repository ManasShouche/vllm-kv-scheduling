"""Materialize exact token prompts and open-loop arrivals once per seed."""
import argparse
import hashlib
import json
import math
import random
from pathlib import Path


POLICY_BUDGETS = {"fixed_512": 512, "fixed_2k": 2048, "fixed_16k": 16384, "ppas": 16384, "decode_guard": 16384, "reserve_static": 512, "reserve_feedback": 512, "cap_2": 512, "evict_cost": 512, "evict_fair": 512}

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def schedule_records_digest(rows):
    """Hash scheduled request metadata independent of completion/write order."""
    keys = ("request_id", "arrival_s", "phase", "warmup", "prompt_tokens", "output_tokens")
    records = [{key: row[key] for key in keys} for row in rows]
    records.sort(key=lambda row: (row["arrival_s"], row["request_id"]))
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def validate(config):
    if config.get("schema_version") != 1:
        raise ValueError("Unsupported configuration schema")
    if not config.get("phases") or not config.get("seeds"):
        raise ValueError("Phases and seeds must be nonempty")
    if len({p["name"] for p in config["phases"]}) != len(config["phases"]):
        raise ValueError("Phase names must be unique")
    if len(set(config["seeds"])) != len(config["seeds"]):
        raise ValueError("Seeds must be unique")
    if not math.isfinite(config["burstiness"]) or config["burstiness"] <= 0:
        raise ValueError("burstiness must be positive")
    total = 0
    for p in config["phases"]:
        for key in ("duration_s", "request_rate", "prompt_tokens", "output_tokens"):
            if not math.isfinite(p[key]) or p[key] <= 0:
                raise ValueError(f"Invalid {key} in {p['name']}")
        if not isinstance(p["prompt_tokens"], int) or not isinstance(p["output_tokens"], int):
            raise ValueError("Token lengths must be integers")
        if p["prompt_tokens"] + p["output_tokens"] >= config["max_model_len"]:
            raise ValueError("Request approaches context boundary; reserve at least one token")
        for key in ('prompt_tokens', 'output_tokens'):
            choices = p.get(key+'_choices', [p[key]])
            if not isinstance(choices, list) or not choices or any(type(n) is not int or n <= 0 for n in choices):
                raise ValueError('Length choices must be nonempty positive integer lists')
        if max(p.get('prompt_tokens_choices', [p['prompt_tokens']])) + max(p.get('output_tokens_choices', [p['output_tokens']])) >= config['max_model_len']:
            raise ValueError('Mixed request approaches context boundary')
        total += p["duration_s"]
    if not 0 <= config["warmup_s"] < config["phases"][0]["duration_s"]:
        raise ValueError("Warmup must fit inside first phase")
    for key in ("request_timeout_s", "drain_timeout_s", "max_active_requests", "max_num_seqs"):
        if config[key] <= 0:
            raise ValueError(f"{key} must be positive")
    for n in config["kv_capacities_bytes"].values():
        if not isinstance(n, int) or n <= 0:
            raise ValueError("KV capacities must be positive bytes")
    if not config["policies"] or any(
            name not in POLICY_BUDGETS or budget != POLICY_BUDGETS[name]
            for name, budget in config["policies"].items()):
        raise ValueError("Unknown policy or incorrect pinned token budget")
    return total


def arrivals(config, seed):
    validate(config)
    rng = random.Random(seed)
    lengths = random.Random(seed + 200000)
    offset = 0.0
    rows = []
    for phase in config["phases"]:
        t = 0.0
        while True:
            t += rng.gammavariate(config["burstiness"], 1 / (phase["request_rate"] * config["burstiness"]))
            if t >= phase["duration_s"]:
                break
            rows.append({"request_id": f"s{seed}-{len(rows):06d}",
                         "arrival_s": offset + t, "phase": phase["name"],
                         "warmup": offset + t < config["warmup_s"],
                         "prompt_tokens": lengths.choice(phase["prompt_tokens_choices"]) if "prompt_tokens_choices" in phase else phase["prompt_tokens"],
                         "output_tokens": lengths.choice(phase["output_tokens_choices"]) if "output_tokens_choices" in phase else phase["output_tokens"]})
        offset += phase["duration_s"]
    return rows


def materialize(config_path, output, model_revision):
    from transformers import AutoTokenizer
    config = json.loads(Path(config_path).read_text())
    validate(config)
    if len(model_revision) != 40 or any(c not in "0123456789abcdef" for c in model_revision):
        raise ValueError("Use a resolved 40-character model commit")
    dest = Path(output)
    dest.mkdir(parents=True, exist_ok=False)
    tok = AutoTokenizer.from_pretrained(config["model"], revision=model_revision, trust_remote_code=False)
    passage = "Explain how queues, memory allocation, and scheduling affect a reliable inference service. "
    pool = tok.encode(passage, add_special_tokens=False)
    (dest / "config.json").write_text(json.dumps(config, indent=2))
    hashes = {}
    for seed in config["seeds"]:
        path = dest / f"seed-{seed}.jsonl"
        rng = random.Random(seed + 100000)
        with path.open("w") as f:
            for row in arrivals(config, seed):
                # Synthetic exact-length tokens, not a claim of natural-text representativeness.
                prefix = tok.encode(f"Experiment {row['request_id']} {rng.getrandbits(64)}. ", add_special_tokens=False)
                count = row["prompt_tokens"]
                row["prompt_token_ids"] = (prefix + pool * (count // len(pool) + 1))[:count]
                assert len(row["prompt_token_ids"]) == count
                f.write(json.dumps(row, separators=(",", ":")) + "\n")
        hashes[path.name] = digest(path)
    manifest = {"model": config["model"], "model_revision": model_revision,
                "config_sha256": digest(dest / "config.json"), "schedule_sha256": hashes,
                "workload_kind": "synthetic exact token lengths", "tokenizer_class": type(tok).__name__}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--model-revision", required=True)
    a = p.parse_args()
    print(json.dumps(materialize(a.config, a.output, a.model_revision), indent=2))


if __name__ == "__main__":
    main()
