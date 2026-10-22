"""Find out whether a Mica server is reachable and, if not, which way to install it on this machine.
Prints a recommendation and the exact commands. Read-only: installs nothing.
  python skills/mica/scripts/mica_setup_check.py [--url http://127.0.0.1:8010]"""
import argparse, json, platform, shutil, subprocess, sys, urllib.request

PROBE = {"state": "Deploy log: canary retried twice after a health-check timeout; p99 latency rose on checkout.",
         "questions": {"q": {"type": "noul", "instructions": "Is checkout degraded?"}}}


def server_ok(url):
    try:
        req = urllib.request.Request(url.rstrip("/") + "/v1/systemone", data=json.dumps(PROBE).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=5) as r:
            return "answers" in json.loads(r.read())
    except Exception:
        return False


def nvidia():
    if not shutil.which("nvidia-smi"):
        return []
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        name, mem, cap = [x.strip() for x in line.split(",")]
        gpus.append({"name": name, "memory_gib": round(float(mem) / 1024, 1), "compute_cap": cap})
    return gpus


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8010")
    a = ap.parse_args()
    if server_ok(a.url):
        print(f"Mica is running at {a.url}. Set MICA_URL={a.url}/v1/systemone in your code.")
        return
    gpus = nvidia()
    os_name = platform.system()
    print("No Mica server at", a.url)
    print("OS:", os_name, platform.machine(), "| Python:", platform.python_version(), "| NVIDIA GPUs:", gpus or "none")
    best = max(gpus, key=lambda g: g["memory_gib"], default=None)
    if best and best["memory_gib"] >= 7.5 and float(best["compute_cap"]) >= 8.0 and os_name in ("Linux", "Windows"):
        ctx = "32k" if best["memory_gib"] >= 23 else "27k" if best["memory_gib"] >= 15.5 else "12k" if best["memory_gib"] >= 11.5 else "6k"
        print(f"\nRecommended: INT8 CUDA runtime on {best['name']} (longest input about {ctx} tokens). Fastest path.")
        if os_name == "Windows":
            print(r"""
Native Windows (experimental), PowerShell, Python 3.10 to 3.13:
git clone https://github.com/akivet/Mica; cd Mica
py -3.12 -m venv .venv; .venv\Scripts\Activate.ps1
pip install -r v0.5/serving/requirements-windows.txt
hf download sky7350/Mica-v0.5-4B --include "int8/*" --local-dir weights
python v0.5/serving/serve.py --backend hf --model weights/int8 --name mica-v0.5-4b --port 8010 --state-dir .mica-state

If its start line says "bfloat16" or requests are slow, use WSL2 (Ubuntu) with the Linux commands below instead;
WSL2 sees the NVIDIA GPU through the Windows driver and runs at full speed.""")
        print(r"""
git clone https://github.com/akivet/Mica && cd Mica
python3 -m venv .venv && . .venv/bin/activate          # Python 3.10 to 3.13
pip install -r v0.5/serving/requirements.txt
pip install --no-build-isolation causal-conv1d==1.7.0
hf download sky7350/Mica-v0.5-4B --include "int8/*" --local-dir weights
python v0.5/serving/serve.py --backend hf --model weights/int8 --name mica-v0.5-4b --port 8010 --state-dir .mica-state
""")
        print("Keep --state-dir on a persistent disk: the first start on a GPU tunes kernels and captures CUDA graphs "
              "and saves them there (about 3 min the first time and 1 min after on an RTX 6000 Ada; CPU-bound, so a "
              "shared, busy host CPU can take over an hour; the server prints each stage). Run it as a long-lived "
              "service, not per call.")
        if len(gpus) > 1:
            print("Several GPUs: pin one with CUDA_VISIBLE_DEVICES=<index> (one server per GPU).")
    elif best and float(best["compute_cap"]) >= 8.0:
        print(f"\n{best['name']} has {best['memory_gib']} GiB: below the INT8 runtime's 8 GB. Use the GGUF path (Docker) with "
              "mica-v0.5-4b-IQ4_XS.gguf, or a larger GPU for the low-latency INT8 path.")
    else:
        print("\nNo suitable NVIDIA GPU: use the GGUF path (llama.cpp) - see README 'Docker (GGUF, llama.cpp)'. "
              "It works on AMD, Apple and CPU but is several times slower than the INT8 CUDA runtime.")


if __name__ == "__main__":
    main()
