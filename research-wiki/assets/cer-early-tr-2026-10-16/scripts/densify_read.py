"""Read the two task-B densification probes (early arm, shadow base) and report memory rows' densification statistics.

Plan: research-wiki/operations/cer-multi4d-early-tr-plan-2026-10-16.md, section 9. Descriptive; no threshold decides.
Usage: python densify_read.py --early <run dir> --shadow <run dir> --out <json> [--from_iter 3100 --to_iter 15900]
Refuses a run unless provenance.txt records exit_code=0, probe/complete.json is complete, both snapshots are populated and
no fine-stage statistics call was unmatched.
"""
import argparse
import json


def load(run):
    prov = dict(l.rstrip("\n").split("=", 1) for l in open(run + "/provenance.txt") if "=" in l)
    assert prov.get("exit_code") == "0", "%s: job exit_code %s" % (run, prov.get("exit_code"))
    assert prov.get("substrate_state") and prov.get("substrate_state") == prov.get("substrate_state_after"), \
        "%s: worktree state changed (%s -> %s)" % (run, prov.get("substrate_state"), prov.get("substrate_state_after"))
    d = run + "/probe"
    c = json.load(open(d + "/complete.json"))
    assert c.get("complete") is True, "%s: probe run not complete" % d
    assert not [k for k in c.get("unmatched", {}) if not k.endswith("/coarse")], "%s: unmatched fine calls %s" % (d, c["unmatched"])
    lab = json.load(open(d + "/labels.json"))
    for k in ("snapshot_labelling", "snapshot_%d" % c["snapshot_iter"]):
        assert k in lab and lab[k]["fg"] is not None and lab[k]["bg"] is not None, "%s: snapshot %s missing" % (d, k)
    recs = [json.loads(l) for l in open(d + "/densify_probe.jsonl") if l.strip()]
    assert recs, "%s: no densify records" % d
    return recs


def snaps(run):
    lab = json.load(open(run + "/probe/labels.json"))
    return {k: lab[k] for k in lab if k.startswith("snapshot_")}


def summarise(recs, bank, lo, hi):
    rs = [r for r in recs if r["bank"] == bank and r["stage"] == "fine" and r["labelled"] and lo <= r["iteration"] <= hi]
    out = {"steps": len(rs)}
    for c in ("0", "1", "2"):
        gs = [r["groups"][c] for r in rs if r["groups"][c]["rows"] > 0]
        if not gs:
            continue
        mean = lambda k: sum(g[k] for g in gs if g[k] is not None) / max(1, sum(1 for g in gs if g[k] is not None))
        out[c] = {"steps_with_rows": len(gs), "grad_mean_seen": mean("grad_mean_seen"), "grad_u_mean_seen": mean("grad_u_mean_seen"),
                  "grad_q50_seen": mean("grad_q50_seen"), "frac_ge_thr_seen": mean("frac_ge_thr_seen"),
                  "denom_mean_seen": mean("denom_mean_seen"),
                  "zero_inc_per_row": sum(g["zero_inc_sum"] for g in gs) / max(1, sum(g["rows"] for g in gs)),
                  "all_gap_per_row": sum(g["all_gap_sum"] for g in gs) / max(1, sum(g["rows"] for g in gs)),
                  "zero_inc_share_of_denom": sum(g["zero_inc_sum"] for g in gs) / max(1e-9, sum(g["denom_sum"] for g in gs)),
                  "mixed_per_row": sum(g["mixed_sum"] for g in gs) / max(1, sum(g["rows"] for g in gs)),
                  "clone": sum(g["clone"] for g in gs), "split": sum(g["split"] for g in gs),
                  "clone_undiluted": sum(g["clone_undiluted"] for g in gs), "split_undiluted": sum(g["split_undiluted"] for g in gs),
                  "rows_first": gs[0]["rows"], "rows_last": gs[-1]["rows"], "iter_last": rs[-1]["iteration"]}
    return out


def ratio(x, y):
    return None if x is None or y in (None, 0) else x / y


ap = argparse.ArgumentParser()
ap.add_argument("--early", required=True)
ap.add_argument("--shadow", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--from_iter", type=int, default=3100)
ap.add_argument("--to_iter", type=int, default=15900, help="background window end")
ap.add_argument("--fg_to_iter", type=int, default=9900, help="foreground window end (density control stops at phase 3)")
a = ap.parse_args()
E, S = load(a.early), load(a.shadow)
rep = {"window": {"bg": [a.from_iter, a.to_iter], "fg": [a.from_iter, a.fg_to_iter]},
       "snapshots": {"early": snaps(a.early), "shadow": snaps(a.shadow)}}
for bank in ("bg", "fg"):
    hi = a.to_iter if bank == "bg" else a.fg_to_iter
    e, s = summarise(E, bank, a.from_iter, hi), summarise(S, bank, a.from_iter, hi)
    rep[bank] = {"early": e, "shadow": s}
    se, ss = rep["snapshots"]["early"].get("snapshot_16000", {}).get(bank), rep["snapshots"]["shadow"].get("snapshot_16000", {}).get(bank)
    if se and ss:
        rep[bank]["rows_at_16000_early_over_shadow"] = {c: ratio(se[c], ss[c]) for c in ("0", "1")}
    if "1" in e and "1" in s:
        rep[bank]["memory_early_over_shadow"] = {k: ratio(e["1"][k], s["1"][k]) for k in
                                                 ("grad_mean_seen", "grad_q50_seen", "frac_ge_thr_seen", "denom_mean_seen",
                                                  "clone", "split")}
        rep[bank]["memory_undiluted_over_diluted_early"] = {
            "grad": ratio(e["1"]["grad_u_mean_seen"], e["1"]["grad_mean_seen"]),
            "clone": ratio(e["1"]["clone_undiluted"], e["1"]["clone"]), "split": ratio(e["1"]["split_undiluted"], e["1"]["split"])}
    if "0" in e and "0" in s:
        rep[bank]["other_early_over_shadow"] = {k: ratio(e["0"][k], s["0"][k]) for k in
                                                ("grad_mean_seen", "clone", "split")}
json.dump(rep, open(a.out, "w"), indent=1)
print(json.dumps(rep, indent=1))
