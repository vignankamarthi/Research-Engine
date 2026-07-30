"""Self-contained TOMATO downloader: urllib + the HF token, per-file timeouts, verbose. Bypasses the
hf library (which hangs on resolve) and ssh-inline quoting. Run on the cluster."""
import json
import os
import sys
import urllib.request

TOK = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip()
DIR = "/scratch/kamarthi_v_neu/tomato"
REPO = "yale-nlp/TOMATO"
os.makedirs(DIR, exist_ok=True)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"Authorization": f"Bearer {TOK}"})
    return urllib.request.urlopen(req, timeout=timeout)


def list_dir(path=""):
    # NON-recursive listing (the ?recursive=true endpoint hangs for this repo)
    url = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{path}".rstrip("/")
    return json.load(get(url, 30))


def main():
    print("listing files (BFS, non-recursive) ...", flush=True)
    files, stack = [], [""]
    while stack:
        d = stack.pop()
        for x in list_dir(d):
            (files.append(x["path"]) if x["type"] == "file" else stack.append(x["path"]))
    print(f"{len(files)} files to fetch", flush=True)
    for i, p in enumerate(files):
        dest = os.path.join(DIR, p)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        url = f"https://huggingface.co/datasets/{REPO}/resolve/main/{p}"
        try:
            with get(url, 120) as r, open(dest, "wb") as f:
                f.write(r.read())
            print(f"[{i+1}/{len(files)}] {os.path.getsize(dest)//1024}KB  {p}", flush=True)
        except Exception as e:
            print(f"[{i+1}/{len(files)}] FAILED {p}: {type(e).__name__} {e}", flush=True)
    total = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(DIR) for f in fs)
    print(f"DONE total {total/1e6:.1f} MB", flush=True)


if __name__ == "__main__":
    sys.exit(main())
