"""GPU scripts require a real EC2 host. They cannot run on an office Mac."""
import json
import platform
import subprocess
import urllib.request


def identity():
    if platform.system() != "Linux":
        raise RuntimeError("GPU execution is permitted only on AWS Linux")
    req = urllib.request.Request("http://169.254.169.254/latest/api/token", method="PUT",
                                 headers={"X-aws-ec2-metadata-token-ttl-seconds": "60"})
    # Bypass HTTP proxies for link-local EC2 metadata; no credentials are requested.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(req, timeout=3) as response:
        token = response.read().decode()
    req = urllib.request.Request("http://169.254.169.254/latest/dynamic/instance-identity/document",
                                 headers={"X-aws-ec2-metadata-token": token})
    with opener.open(req, timeout=3) as response:
        doc = json.load(response)
    if not doc.get("instanceId", "").startswith("i-"):
        raise RuntimeError("EC2 identity was not confirmed")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                                   "--format=csv,noheader"], text=True).strip()
    # Keep account identifiers out of the artifact.
    return {"region": doc["region"], "instance_type": doc["instanceType"],
            "instance_id": doc["instanceId"], "gpu": gpu}


if __name__ == "__main__":
    print(json.dumps(identity(), indent=2))
