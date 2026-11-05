"""Compatibility report for the CER reducer (2026-10-07): runs the INSTALLED reducer's command line on the real
2026-10-02 Ex4DGS and Multi4D scores, compares each written file with the reading of record, and writes a retained report.

Comparison: recursive over every key and list element; floats equal within 1e-12 relative (max(1, |a|)); every other
value exactly equal. Excluded fields (not results): tool, tool_sha256, policy_record, and the new
nonfinite_descriptive_replaced (reported separately). Usage: python reduce_compat_report.py --out <report.json>
"""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys

A = "/leonardo_work/EUHPC_D36_068/sri/proj_adags/agent-control/ars"
EXCLUDED = ("tool", "tool_sha256", "policy_record", "nonfinite_descriptive_replaced")
CASES = [("multi4d", ["--substrate", "multi4d"], "cer_reduce_multi4d.json"),
         ("ex4dgs", ["--substrate", "ex4dgs", "--ex4dgs_base", "replacement"], "cer_reduce_ex4dgs.json")]


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def strip(o):
    if isinstance(o, dict):
        return {k: strip(v) for k, v in o.items() if k not in EXCLUDED}
    if isinstance(o, list):
        return [strip(v) for v in o]
    return o


def diff(a, b, path=""):
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append("%s.%s only in %s" % (path, k, "new" if k in b else "record"))
            else:
                out += diff(a[k], b[k], "%s.%s" % (path, k))
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append("%s length %d != %d" % (path, len(a), len(b)))
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += diff(x, y, "%s[%d]" % (path, i))
    elif isinstance(a, float) and isinstance(b, float):
        if abs(a - b) > 1e-12 * max(1.0, abs(a)):
            out.append("%s %r != %r" % (path, a, b))
    elif a != b or type(a) is not type(b):
        out.append("%s %r != %r" % (path, a, b))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    od = os.path.join(os.path.dirname(os.path.abspath(a.out)), "outputs")
    os.makedirs(od, exist_ok=True)
    import numpy
    import scipy
    rep = {"reducer": os.path.join(A, "cer_reduce.py"), "reducer_sha256": sha(os.path.join(A, "cer_reduce.py")),
           "this_script_sha256": sha(os.path.abspath(__file__)), "python": sys.version.split()[0],
           "numpy": numpy.__version__, "scipy": scipy.__version__, "host": platform.node(),
           "comparison": "recursive; floats within 1e-12 relative (max(1,|a|)); others exact incl. type",
           "excluded_fields": list(EXCLUDED), "cases": {}}
    for name, args, record in CASES:
        out = os.path.join(od, "%s_new.json" % name)
        if os.path.exists(out):
            os.remove(out)
        cmd = [sys.executable, os.path.join(A, "cer_reduce.py"), "--scores", os.path.join(A, "scores", "wave1")] + args + \
              ["--policy_record", "plan log 2026-10-01 evening", "--out", out]
        p = subprocess.run(cmd, capture_output=True, text=True)
        rec_p = os.path.join(A, "checks", "wave1", record)
        case = {"command": cmd, "exit_code": p.returncode, "stdout": p.stdout, "stderr_tail": p.stderr[-2000:],
                "record": rec_p, "record_sha256": sha(rec_p)}
        if p.returncode == 0:
            new = json.load(open(out))
            case.update({"output": out, "output_sha256": sha(out),
                         "nonfinite_descriptive_replaced": new.get("nonfinite_descriptive_replaced"),
                         "inputs_sha256_identical": new.get("inputs") == json.load(open(rec_p)).get("inputs")})
            d = diff(strip(json.load(open(rec_p))), strip(new))
            case.update({"n_differences": len(d), "differences": d[:50]})
        rep["cases"][name] = case
    json.dump(rep, open(a.out, "w"), indent=1)
    print(json.dumps({k: {kk: v[kk] for kk in ("exit_code", "n_differences", "inputs_sha256_identical",
                                                "nonfinite_descriptive_replaced") if kk in v}
                      for k, v in rep["cases"].items()}))
    print("reducer", rep["reducer_sha256"], "report", a.out)


if __name__ == "__main__":
    main()
