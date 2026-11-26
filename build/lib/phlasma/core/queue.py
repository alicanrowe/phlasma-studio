"""phlasma batch queue.txt -j N : each line '<module> <verb> <job> --out <dir> [...]'. N jobs at once, one process each."""
import argparse, json, shlex, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor


def main(argv):
    ap = argparse.ArgumentParser(prog="phlasma batch")
    ap.add_argument("queue")
    ap.add_argument("-j", type=int, default=4)
    a = ap.parse_args(argv)
    lines = [l.strip() for l in open(a.queue) if l.strip() and not l.startswith("#")]

    def one(line):
        t = time.time()
        p = subprocess.run([sys.executable, "-m", "phlasma", *shlex.split(line), "--json"], capture_output=True, text=True)
        try:
            r = json.loads(p.stdout.strip().splitlines()[-1])
        except Exception:
            r = {"status": "error", "stderr": p.stderr[-2000:]}
        r["line"], r["wall"] = line, round(time.time() - t, 1)
        print(json.dumps({"ev": "job_done", "line": line, "status": r.get("status"), "wall": r["wall"]}), flush=True)
        return r
    with ThreadPoolExecutor(a.j) as ex:
        res = list(ex.map(one, lines))
    bad = [r for r in res if r.get("status") != "ok"]
    print(json.dumps({"ev": "batch_done", "jobs": len(res), "failed": len(bad)}))
    return 0 if not bad else 3
