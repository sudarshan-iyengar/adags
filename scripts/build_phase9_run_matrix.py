#!/usr/bin/env python3
"""Generate and validate the preregistered Phase 9 execution DAG."""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path

SCENES=("coffee_martini","cook_spinach","cut_roasted_beef","flame_salmon_1","flame_steak","sear_steak")
ANN=("cut_roasted_beef","flame_steak","sear_steak")
MODES=("route0","capacity_only","visibility_only","full","shuffled")
B03_MODES=("route0","null_reset","capacity_only","oracle_capacity","visibility_only","full","shuffled")
ROOT="$WORK/proj_adags/runs/phase9-depth-visibility-capacity/cycle-v1"
RES={
"login":("none",None,1,0,4,15),
"external":("none",None,0,0,0,0),
"cpu30":("slurm","boost_usr_prod",8,0,32,30),
"cpu2":("slurm","boost_usr_prod",8,0,64,120),
"cpu4":("slurm","boost_usr_prod",16,0,128,240),
"cpu12":("slurm","boost_usr_prod",32,0,384,720),
"gpu30":("slurm","boost_usr_prod",8,1,64,30),
"gpu2":("slurm","boost_usr_prod",8,1,64,120),
"gpu4":("slurm","boost_usr_prod",8,1,64,240),
"gpu15":("slurm","boost_usr_prod",8,1,64,900),
"gpu24":("slurm","boost_usr_prod",8,1,64,1440),
}
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(v): return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def od(run): return f"{ROOT}/executions/{run}"
def out(schema,path): return {"schema":schema,"path":path,"sha256":None,"required":True}
def ref(run,schema,path): return {"producer_run_id":run,"schema":schema,"path":path,"sha256":None,"status":"unresolved_until_producer"}
def ck(scene,seed,lane,it): return f"{ROOT}/{scene}/seed-{seed}/{lane}/chkpnt{it}.pth"
def sched(profile,run,deps):
    kind,part,cpus,gpus,mem,mins=RES[profile]
    return {"profile":profile,"site":"leonardo" if kind=="slurm" else profile,
      "scheduler":kind,"partition":part,"account":"euhpc_d21_034" if part else None,
      "qos":"boost_qos_lprod" if part else None,"nodes":1 if cpus else 0,
      "tasks_per_node":1 if cpus else 0,"cpus_per_task":cpus,"gpus_per_node":gpus,
      "gpu_type":"a100" if gpus else None,"gpu_memory_gib":64 if gpus else None,
      "host_memory_gib":mem,"wall_minutes":mins,
      "dependency_mode":"afterok" if part and deps else None,
      "stdout":f"logs/{run}_%j.out" if part else None,
      "stderr":f"logs/{run}_%j.err" if part else None}
def add(runs,run,stage,action,claim,need,scope,profile,deps=(),scene=None,pop="not_applicable",
        seed=None,k=None,start=None,end=None,mode=None,input_run=None,input_path=None,
        outputs=None,promote="",fail="",conditional=False,external=None,temp=4,durable=4,ckg=0):
    deps=tuple(deps); s=sched(profile,run,deps)
    argv=["python","scripts/run_phase9_depth_visibility.py",action,"--run-id",run,
          "--execution-manifest",f"{od(run)}/resolved-execution.json"]
    if scene: argv+=["--scene",scene]
    training=None
    if start is not None:
        training={"mode":mode,"start_iteration":start,"end_iteration":end,
          "point_ceiling":600000,"requested_k":k,"topology_cutoff_iteration":5000,
          "input_checkpoint":None if input_run is None else ref(input_run,"adags-checkpoint-v1",input_path),
          "output_checkpoint":{"schema":"adags-checkpoint-v1","path":ck(scene,seed,mode,end),"sha256":None},
          "run_directory":od(run),"capacity_seed":seed,"transaction_id":None,
          "transaction_state":None,"source_wandb_run_id":None,"output_wandb_run_id":None}
    item={"schema_version":"phase9-run-entry-v2","run_id":run,"stage":stage,"action":action,
      "claim_or_uncertainty":claim,"necessity":need,"scope":scope,"conditional":conditional,
      "scene":scene,"camera_time_population":pop,
      "seeds":{"training":seed,"capacity":seed,"shuffle":None if seed is None else seed+17011,
               "bootstrap":20260715 if "score" in action or "decide" in action else None},
      "prerequisites":list(deps),"external_inputs":external or [],
      "input_artifacts":[ref(d,"phase9-terminal-manifest-v1",f"{od(d)}/terminal.json") for d in deps],
      "implementation_binding":{"producer_run_id":"P9-I01-IMPLEMENTATION-FREEZE",
        "required_before_submission":s["scheduler"]=="slurm","resolved_manifest_sha256":None,
        "status":"not_applicable" if run=="P9-I01-IMPLEMENTATION-FREEZE" else "unresolved"},
      "command":{"argv_template":argv,"argv_template_sha256":h(argv),"resolved_argv":None,
        "resolved_argv_sha256":None,"launcher_path":"scripts/run_phase9_depth_visibility.py",
        "launcher_sha256":None,"status":"requires_I01_resolution"},
      "configuration":{"objective":"research-wiki/objectives/depth-visibility-capacity-v1.md",
        "method":"research-wiki/operations/phase9-csvl-isr-v1-method.md",
        "slice_b_contract":"research-wiki/operations/phase9-csvl-isr-v1-slice-b-contract.md",
        "split":"configs/depth_visibility/n3v_split_v1.json",
        "annotation_windows":"configs/depth_visibility/annotation_windows_v1.json",
        "base_config_path":"configs/depth_visibility/csvl_isr_v1.json",
        "base_config_sha256":sha(Path(__file__).resolve().parents[1]/"configs/depth_visibility/csvl_isr_v1.json"),"derived_config_path":"configs/depth_visibility/isr_modes_v1.json" if training is not None else None,"derived_config_sha256":sha(Path(__file__).resolve().parents[1]/"configs/depth_visibility/isr_modes_v1.json") if training is not None else None,
        "resolved_merged_config_sha256":None,"status":"requires_I01_resolution"},
      "training":training,"scheduler":s,"max_gpu_hours":s["gpus_per_node"]*s["wall_minutes"]/60,
      "storage":{"maximum_temporary_gib":temp,"maximum_durable_run_gib":durable,
        "maximum_checkpoint_output_gib":ckg,"output_root":od(run),
        "retention_class":"transient_generated" if s["scheduler"]=="slurm" else "durable_decision"},
      "expected_outputs":[out("phase9-terminal-manifest-v1",f"{od(run)}/terminal.json")]+(outputs or []),
      "completion_rule":"Slurm terminal state, exit 0, and all required hashes" if s["scheduler"]=="slurm" else "validated output and all required hashes",
      "pre_observation_promotion_predicate":promote,
      "failure_interpretation_and_action":fail,"status":"registered_not_submitted"}
    if action=="score-gate-a":
        item["predicate_binding"]=predicate_binding(run,action)
    runs.append(item); return run
def predicate_binding(run,action):
    mapping={
      "P9-I00-CODE-REVIEW-DECISION":("code_review_admission_v1",None),
      "P9-A06-CUT-ENGINEERING-DECISION":("gate_a_engineering_v1","gate_a.engineering"),
      "P9-A06-CLAIM-GRADE-DECISION":("gate_a_claim_grade_v1","gate_a.claim_grade"),
      "P9-B00-SIDECAR-EVAL-CONFORMANCE":("b00_conformance_v1","representation"),
      "P9-B01-CUT-S0-PILOT-DECISION":("b01_pilot_stability_v1","representation"),
      "P9-B02-CUT-S0-ORACLE-DECISION":("b02_oracle_viability_v1","gate_b"),
      "P9-B02-MOMENT-REPAIR-TRIGGER-DECISION":("b02_moment_trigger_v1","representation"),
      "P9-B02-REPAIR-DECISION":("b02_repair_resolution_v1","representation"),
      "P9-B02-CAPACITY-ADMISSION-DECISION":("b02_capacity_admission_v1","representation"),
      "P9-B03-CUT-S0-GATE-DECISION":("gate_b_cut_b03_v1","gate_b"),
      "P9-B04-S0-ANALYZE-DECISION":("gate_b_seed0_expansion_v1","gate_b"),
      "P9-B04-SEEDS-ANALYZE-DECISION":("gate_b_selected_seeds_v1","gate_b"),
    }
    if run in mapping: pid,key=mapping[run]
    elif action=="score-gate-a":
        pid,key=("gate_a_engineering_score_v1","gate_a.engineering") if "CUT" in run else ("gate_a_claim_grade_score_v1","gate_a.claim_grade")
    else: pid,key=("execution_completion_v1",None)
    repo=Path(__file__).resolve().parents[1]
    return {"predicate_id":pid,"expression_config_key":key,
      "scientific_config_path":"configs/depth_visibility/csvl_isr_v1.json",
      "scientific_config_sha256":sha(repo/"configs/depth_visibility/csvl_isr_v1.json"),
      "contract_path":"research-wiki/operations/phase9-csvl-isr-v1-slice-b-contract.md",
      "contract_sha256":sha(repo/"research-wiki/operations/phase9-csvl-isr-v1-slice-b-contract.md")}

def decide(runs,run,stage,claim,deps,promote,fail,scope="engineering",profile="cpu4",conditional=False):
    add(runs,run,stage,"decide",claim,"Produces a typed decision; no downstream free-string alias.",scope,profile,deps,
      outputs=[out("phase9-decision-v1",f"{od(run)}/decision.json"),out("phase9-provenance-v1",f"{od(run)}/provenance.json")],
      promote=promote,fail=fail,conditional=conditional)
    runs[-1]["predicate_binding"]=predicate_binding(run,"decide")
    return run

def train(runs,run,stage,scene,seed,mode,start,end,k,deps,claim,promote,fail,input_run=None,profile="gpu4",conditional=False):
    deps=list(deps)
    specialized=[]
    train_side=f"P9-P04-{scene.upper()}-TRAIN-SIDECARS-FREEZE"
    eval_freeze=f"P9-P05-{scene.upper()}-EVALUATOR-FREEZE"
    if mode in {"null_reset","capacity_only","visibility_only","full","shuffled"}:
        if train_side not in deps: deps.append(train_side)
        specialized.append(ref(train_side,"phase9-train-sidecars-v1",f"{ROOT}/sidecars/{scene}/train-freeze.json"))
    if mode in {"oracle_capacity","oracle_capacity_medianmom"}:
        oracle="P9-P05-CUT-ORACLE-SIDECAR-FREEZE"
        if oracle not in deps: deps.append(oracle)
        specialized.append(ref(oracle,"phase9-oracle-capacity-sidecar-v1",f"{ROOT}/sidecars/cut_roasted_beef/oracle.json"))
    if end==6000:
        if eval_freeze not in deps: deps.append(eval_freeze)
        specialized.append(ref(eval_freeze,"phase9-evaluator-freeze-v1",f"{ROOT}/evaluation/{scene}/freeze.json"))
    if stage in {"B03","B04","B04-seeds"} and mode in {"null_reset","capacity_only","oracle_capacity","full","shuffled"}:
        policy="P9-B02-CAPACITY-ADMISSION-DECISION"
        if policy not in deps: deps.append(policy)
        specialized.append(ref(policy,"phase9-optimizer-policy-v1",f"{od(policy)}/optimizer-policy.json"))
    add(runs,run,stage,"train",claim,"Smallest registered lane isolating its causal factor.",
      "claim_grade" if end==6000 else "engineering",profile,deps,scene,
      "frozen transforms_train; frozen transforms_test/evaluator",seed,k,start,end,mode,
      input_run,None if input_run is None else ck(scene,seed,"common",5000),
      [out("adags-checkpoint-v1",ck(scene,seed,mode,end)),
       out("phase9-training-metrics-v1",f"{od(run)}/metrics.json"),
       out("phase9-capacity-ledger-v1",f"{od(run)}/capacity-ledger.json"),
       out("phase9-provenance-v1",f"{od(run)}/provenance.json"),
       out("phase9-render-inventory-v1",f"{od(run)}/renders.json")],
      promote,fail,conditional,temp=20,durable=50,ckg=20)
    runs[-1]["input_artifacts"].extend(specialized)
    runs[-1]["predicate_binding"]=predicate_binding(run,"train")
    return run
def build(repo):
    runs=[]
    add(runs,"P9-A00-STATIC-S20260715","A00","static","Schemas, camera math, geometry, metrics and failures are executable.","Cheap admission.","engineering","login",seed=20260715,promote="Pass twice with equal canonical hashes.",fail="Correct code; no Slurm.")
    add(runs,"P9-A01-SYNTH-S20260715","A01","synthetic","Controlled order/reveal succeeds and corruptions fail.","Separates math from real data.","engineering","login",("P9-A00-STATIC-S20260715",),seed=20260715,promote="All positives and corruptions behave exactly.",fail="Correct code; no Slurm.")
    decide(runs,"P9-I00-CODE-REVIEW-DECISION","I00","Independent xhigh code review has no critical defect.",("P9-A00-STATIC-S20260715","P9-A01-SYNTH-S20260715"),"Critical fixed; high fixed or fail-closed.","Block checkpoint.","engineering","login")
    add(runs,"P9-I01-IMPLEMENTATION-FREEZE","I01","freeze-implementation","Committed/pushed code, configs, commands and environment are immutable.","Every Slurm run needs exact bindings.","engineering","login",("P9-I00-CODE-REVIEW-DECISION",),
      outputs=[out("phase9-implementation-freeze-v1",f"{ROOT}/authority/implementation-freeze.json"),out("phase9-command-registry-v1",f"{ROOT}/authority/commands.json")],
      promote="Clean tracked branch/upstream; all hashes resolve.",fail="Repair, re-review, commit, push; no submit.")
    add(runs,"P9-A02-DA3-WEIGHT-SHA","A02","hash-da3","Exact read-only DA3 authority is sealed.","First-observed weight digest must be measured.","engineering","cpu30",("P9-I01-IMPLEMENTATION-FREEZE",),
      outputs=[out("phase9-da3-authority-v1",f"{ROOT}/authority/da3.json")],
      promote="Completed/0:0; size, source, model, job, command and SHA seal.",fail="Block DA3.",temp=8,durable=1)
    add(runs,"P9-A03-DA3-CONFORMANCE-S20260715","A03","da3-conformance","K/w2c, optical-z, resize, ancestry and numeric repeats conform.","Prevents coordinate/scale defects.","engineering","gpu30",("P9-A02-DA3-WEIGHT-SHA",),"cut_roasted_beef","analytic plus cut frame0 groups",20260715,
      promote="All tolerances pass; both raw hashes recorded; numeric agreement 1e-5.",fail="Block production inference.",temp=20,durable=10)
    add(runs,"P9-A04-CUT-F0125-0127-S20260715","A04","tiny-csvl","Target-free geometry gives finite support and correct controls.","Small real-data diagnostic.","exploratory","gpu2",("P9-A03-DA3-CONFORMANCE-S20260715",),"cut_roasted_beef","frames125-127 all train cameras; cam00 target",20260715,
      promote="No target ancestry; finite nonzero support; controls and sealed-sidecar repeat pass.",fail="Diagnose once; block full ledgers.",temp=30,durable=20)
    add(runs,"P9-A05-ANNOTATION-PACKET","A05","build-annotation-packet","Blinded 54-window packet has raw RGB only and empty labels.","Real Gate A needs genuine humans.","engineering","cpu2",("P9-I01-IMPLEMENTATION-FREEZE",),pop="54 frozen windows",
      outputs=[out("phase9-annotation-packet-v1",f"{ROOT}/annotation/packet-manifest.json"),out("phase9-r009-separation-proof-v1",f"{ROOT}/annotation/r009-separation.json")],
      promote="Schema/hash/R009-margin/two-role checks; distinct IDs before open; fields empty.",fail="Correct packet; never fabricate labels.",temp=50,durable=20)

    p03={}; side={}; labels={}; evaluators={}
    for scene in SCENES:
        p1=f"P9-P01-{scene.upper()}-DA3-PRED-S20260715"; p2=f"P9-P02-{scene.upper()}-FLOW-ADAPT-S20260715"
        p3=f"P9-P03-{scene.upper()}-CSVL-LEDGER-S20260715"; p4=f"P9-P04-{scene.upper()}-TRAIN-SIDECARS-FREEZE"
        add(runs,p1,"P01","produce-da3","All target-conditioned calibrated depth groups exist.","Three-frame diagnostic is insufficient.","engineering","gpu24",("P9-A04-CUT-F0125-0127-S20260715",),scene,
          "all 300 times; all valid train groups; ancestry supports cam00 and LOCO train targets",20260715,
          outputs=[out("phase9-da3-sidecar-v1",f"{ROOT}/preprocess/{scene}/da3/manifest.json"),out("phase9-da3-array-inventory-v1",f"{ROOT}/preprocess/{scene}/da3/arrays.json")],
          promote="Complete exact arrays/K/w2c/group/ancestry hashes seal.",fail="No partial scoring; classify failure.",temp=250,durable=200)
        add(runs,p2,"P02","adapt-flow","Existing flow has explicit direction, validity and hashes.","Temporal/flicker need trustworthy correspondence.","engineering","cpu12",("P9-I01-IMPLEMENTATION-FREEZE",),scene,"all existing scene flow NPZ and image pairs",20260715,
          outputs=[out("depth-visibility-flow-schema-v1",f"{ROOT}/preprocess/{scene}/flow/manifest.json")],
          promote="Every used file passes schema/translation/cycle/source/array checks.",fail="Register new generation cycle; never guess direction.",temp=150,durable=20)
        add(runs,p3,"P03","build-csvl","Complete label-free calibrated visibility ledger exists.","Gate A and sidecars need sealed predictions.","engineering","cpu12",(p1,p2),scene,"all registered target cameras/times",20260715,
          outputs=[out("phase9-csvl-ledger-v1",f"{ROOT}/preprocess/{scene}/csvl/ledger.json"),out("phase9-csvl-array-inventory-v1",f"{ROOT}/preprocess/{scene}/csvl/arrays.json")],
          promote="Complete target ancestry/provenance/risk/state/track schemas and hash.",fail="Correct geometry before labels/renders.",temp=200,durable=150)
        add(runs,p4,"P04","freeze-train-sidecars","Inferred/weight/shuffle/generic/donor sidecars are immutable and leakage checked.","Training sidecars precede outcomes.","engineering","cpu4",(p3,),scene,"train only; no cam00 RGB/labels/eval/R009",0,
          outputs=[out("phase9-train-sidecars-v1",f"{ROOT}/sidecars/{scene}/train-freeze.json")],
          promote="Read prohibitions, K feasibility, universe and sidecar hashes validate.",fail="Block scene training.",temp=80,durable=50)
        p03[scene]=p3; side[scene]=p4
    for scene in ANN:
        rid=f"P9-A05-{scene.upper()}-LABEL-FREEZE"; labels[scene]=rid
        add(runs,rid,"A05-labels","freeze-human-labels","Genuine two-stage labels/adjudication are immutable.","Only human bridge to gates.","claim_grade","external",("P9-A05-ANNOTATION-PACKET",),scene,"all frozen scene windows",
          outputs=[out("phase9-human-label-freeze-v1",f"{ROOT}/annotation/{scene}/labels.json")],
          external=[{"schema":"phase9-human-annotation-return-v1","path":None,"sha256":None,"status":"awaiting_genuine_human_input","fabrication_prohibited":True}],
          promote="Distinct humans; sealed discovery; union roster; two responses; adjudication/unknown valid.",fail="Remain not_evaluable; continue label-free only.")
    for scene in SCENES:
        rid=f"P9-P05-{scene.upper()}-EVALUATOR-FREEZE"; deps=[f"P9-P02-{scene.upper()}-FLOW-ADAPT-S20260715"]; pop="label-free global/static/flow; event not_evaluable"
        if scene in ANN: deps.append(labels[scene]); pop="frozen human event/static/flow evaluator"
        add(runs,rid,"P05","freeze-evaluator","Masks, LPIPS, flow, formulas and aggregation freeze before renders.","Prevents evaluator tuning.","claim_grade","cpu4",deps,scene,pop,20260715,
          outputs=[out("phase9-evaluator-freeze-v1",f"{ROOT}/evaluation/{scene}/freeze.json")],
          promote="Fixtures, sources, labels, denominators and LPIPS pins validate.",fail="Block scoring; never synthesize labels.",temp=40,durable=20)
        evaluators[scene]=rid
    oracle="P9-P05-CUT-ORACLE-SIDECAR-FREEZE"
    add(runs,oracle,"P05","freeze-oracle-sidecar","Oracle uses visible train polygons and calibrated depth only.","Mandatory causal attribution.","engineering","cpu4",(p03["cut_roasted_beef"],labels["cut_roasted_beef"]),"cut_roasted_beef","visible train surfaces; cam00 only event definition",0,
      outputs=[out("phase9-oracle-capacity-sidecar-v1",f"{ROOT}/sidecars/cut_roasted_beef/oracle.json")],
      promote="No cam00 xyz/color; source polygon/K/w2c/DA3/track hashes; K feasible.",fail="Oracle not_evaluable; block coupling.",temp=40,durable=20)
    baseline={}
    for scene in ANN:
        rid=f"P9-A06-{scene.upper()}-MONOCULAR-BASELINES-S20260715"; baseline[scene]=rid
        add(runs,rid,"A06-baseline","run-monocular-baselines","R031/R032/R033 and R031-MT are frozen before labels on the entire candidate population.","Matched baselines cannot be synthesized in scorer.","claim_grade","gpu15",("P9-A03-DA3-CONFORMANCE-S20260715","P9-A05-ANNOTATION-PACKET"),scene,"all frozen windows; cam00 plus every transforms_train camera; target-consuming; labels unopened",20260715,
          outputs=[out("phase9-r031-family-predictions-v1",f"{ROOT}/gate-a/{scene}/baselines.json")],
          promote="Historical code/command pins and exact prediction/source hashes seal.",fail="Baseline-relative Gate A not_evaluable.",temp=100,durable=60)
    cutscore="P9-A06-CUT-CALIBRATE-SCORE"
    add(runs,cutscore,"A06-cut","score-gate-a","Cut calibration freezes estimators and development scores once.","Approved method-development gate.","engineering","cpu4",(p03["cut_roasted_beef"],baseline["cut_roasted_beef"],labels["cut_roasted_beef"]),"cut_roasted_beef","7 calibration + 11 development windows",20260715,
      outputs=[out("phase9-gate-a-score-v1",f"{ROOT}/gate-a/cut_roasted_beef/score.json"),out("phase9-gate-a-calibrator-v1",f"{ROOT}/gate-a/cut_roasted_beef/calibrator.json")],
      promote="Every engineering conjunction passes with exact hashes.",fail="Fail/not_evaluable blocks B03; do not tune.")
    cutgate=decide(runs,"P9-A06-CUT-ENGINEERING-DECISION","A06-cut-decision","CSVL passes frozen cut engineering Gate A.",(cutscore,),"All exact criteria pass; emit method/baseline/calibrator/population hashes.","Block B03; label-free work remains exploratory.")
    transfers=[]
    for scene in ("flame_steak","sear_steak"):
        rid=f"P9-A06-{scene.upper()}-TRANSFER-SCORE"; transfers.append(rid)
        add(runs,rid,"A06-transfer","score-gate-a","Frozen cut method transfers without retuning.","Each locked scene must pass.","claim_grade","cpu4",(cutgate,p03[scene],baseline[scene],labels[scene]),scene,"all frozen transfer windows",20260715,
          outputs=[out("phase9-gate-a-score-v1",f"{ROOT}/gate-a/{scene}/score.json"),out("phase9-provenance-v1",f"{ROOT}/gate-a/{scene}/provenance.json")],
          promote="Every claim-grade conjunction and represented event family passes.",fail="Claim fails/not_evaluable; no retune.")
    decide(runs,"P9-A06-CLAIM-GRADE-DECISION","A06-transfer-decision","Both locked scenes pass claim-grade Gate A.",transfers,"Both exact transfer decisions pass.","Record partial/negative; new cycle for revision.","claim_grade")
    add(runs,"P9-B00-OPERATOR-STATIC-S0","B00","operator-static","Row surgery, seeds, budgets, metrics and recovery are exact.","Cheap representation admission.","engineering","login",("P9-I01-IMPLEMENTATION-FREEZE",),seed=0,promote="Exact tests pass twice with equal hashes.",fail="Correct; no train.")
    add(runs,"P9-B00-OPERATOR-GPU-SMOKE-S0","B00","operator-gpu-smoke","K8 device transaction has finite render/gradients and exact budget.","Required before common training.","engineering","gpu30",("P9-B00-OPERATOR-STATIC-S0",),seed=0,promote="Finite, exact counts, zero invariants, restart equal.",fail="Block training.",temp=20,durable=10)
    b00=decide(runs,"P9-B00-SIDECAR-EVAL-CONFORMANCE","B00-decision","Operator, sidecar read sets and evaluator aliases conform.",("P9-B00-OPERATOR-GPU-SMOKE-S0",side["cut_roasted_beef"],evaluators["cut_roasted_beef"]),"All schemas/fixtures/prohibited reads/LPIPS pins pass.","Block representation training.")
    common="P9-B01-CUT-S0-COMMON-I5000"
    train(runs,common,"B01","cut_roasted_beef",0,"common",0,5000,None,(b00,side["cut_roasted_beef"]),"Matched iteration-5000 source checkpoint.","Checkpoint/config/seed/budgets validate.","Diagnose base/infrastructure.",None,"gpu15")
    pilots=[]
    for mode in ("route0","capacity_only"):
        rid=f"P9-B01-CUT-S0-{mode.upper()}-I5250"; pilots.append(rid)
        train(runs,rid,"B01","cut_roasted_beef",0,mode,5001,5250,None if mode=="route0" else 256,(common,),
          "Matched route0 or event-blind capacity-only pilot.","Finite exact K/budget; no catastrophic early harm.","Attribute base versus capacity.",common,"gpu2")
    pilotdec=decide(runs,"P9-B01-CUT-S0-PILOT-DECISION","B01-decision","Route0 and capacity pilots establish feasibility.",pilots,"Both satisfy exact stability and emit hashes.","Block oracle; attribute.")
    oraclepilot="P9-B02-CUT-S0-ORACLE-I5250"
    train(runs,oraclepilot,"B02","cut_roasted_beef",0,"oracle_capacity",5001,5250,256,(pilotdec,*pilots,oracle,cutgate),
      "Genuine oracle evidence with identical capacity operator.","Finite exact K/budget; favorable event direction/static safety.","Diagnose sidecar/budget/optimization/operator.",common,"gpu2")
    oracledec=decide(runs,"P9-B02-CUT-S0-ORACLE-DECISION","B02-decision","Oracle capacity is causally viable.",(*pilots,oraclepilot),"Registered directional/static predicates pass.","Attribute failure; only registered repair may follow.")
    trigger=decide(runs,"P9-B02-MOMENT-REPAIR-TRIGGER-DECISION","B02-repair-trigger","Sole moment-repair predicate is evaluated.",(*pilots,oraclepilot),"Pass only if every K256 capacity mode is finite pre-mutation and >2x route0 loss at 5002-5011.","False emits a terminal not_applicable branch outcome.","engineering","cpu4",False)
    runs[-1]["expected_outputs"].append(out("phase9-branch-outcome-v1",f"{od(trigger)}/repair-branch.json"))
    repairs=[]
    for mode in ("capacity_only","oracle_capacity"):
        rid=f"P9-B02-CUT-S0-{mode.upper()}-MEDIANMOM-I5250"; repairs.append(rid); deps=[trigger,common]
        if mode=="oracle_capacity": deps += [oracle,cutgate]
        train(runs,rid,"B02-repair","cut_roasted_beef",0,f"{mode}_medianmom",5001,5250,256,deps,
          "Single coordinate-wise lower-median moment repair.","Same geometry/K/operator/trigger/horizon; spike removed.","Pivot; no second repair.",common,"gpu2",True)
    repairdec=decide(runs,"P9-B02-REPAIR-DECISION","B02-repair-decision","Repair is not_applicable or resolves sole pathology.",(trigger,),"Emit not_applicable if trigger is false; if true, require both registered repair artifacts and both pass.","Pivot operator.","engineering","cpu4",False)
    runs[-1]["conditional_prerequisite_sets"]=[{"when":"repair_branch_equals_run_repairs","required_run_ids":list(repairs)},{"when":"repair_branch_equals_not_applicable","required_run_ids":[]}]
    runs[-1]["conditional_input_artifacts"]=[
      {"when":"repair_branch_equals_run_repairs","artifacts":[ref(r,"phase9-terminal-manifest-v1",f"{od(r)}/terminal.json") for r in repairs]},
      {"when":"repair_branch_equals_not_applicable","artifacts":[ref(trigger,"phase9-branch-outcome-v1",f"{od(trigger)}/repair-branch.json")]}]
    b02=decide(runs,"P9-B02-CAPACITY-ADMISSION-DECISION","B02-admission","One optimizer-state policy admits B03.",(oracledec,repairdec),"Oracle viability under zero or sole repair; emit exactly zero or coordinatewise_lower_median_v1.","Block B03.")
    runs[-1]["expected_outputs"].append(out("phase9-optimizer-policy-v1",f"{od(b02)}/optimizer-policy.json"))
    b03ids=[]
    for mode in B03_MODES:
        rid=f"P9-B03-CUT-S0-{mode.upper()}-I6000"; b03ids.append(rid)
        deps=[common,cutgate,b02,side["cut_roasted_beef"],evaluators["cut_roasted_beef"]]
        if mode=="oracle_capacity": deps.append(oracle)
        train(runs,rid,"B03","cut_roasted_beef",0,mode,5001,6000,2048 if mode in {"null_reset","capacity_only","oracle_capacity","full","shuffled"} else None,deps,
          "Frozen cut causal endpoint for one mode.","Terminal exact artifact; matrix decision applies Gate B.","Scientific failure remains failure.",common,"gpu4")
    b03=decide(runs,"P9-B03-CUT-S0-GATE-DECISION","B03-decision","Full coupling passes cut Gate B and causal controls.",(*b03ids,evaluators["cut_roasted_beef"],cutgate),
      "All route0/control/static/budget/quality predicates; freeze method/config/sidecar/evaluator.","Attribute and block B04.","claim_grade")
    seed0=[]
    for scene in SCENES:
        if scene=="cut_roasted_beef":
            for mode in MODES:
                source=f"P9-B03-CUT-S0-{mode.upper()}-I6000"; rid=f"P9-B04-CUT-S0-{mode.upper()}-REUSE"; seed0.append(rid)
                add(runs,rid,"B04-reuse","verify-reuse","Exact B03 cut artifact is reused.","Avoid duplicate training.","claim_grade","login",(b03,source),scene,"exact B03 artifact",0,
                  outputs=[out("phase9-reuse-verification-v1",f"{od(rid)}/reuse.json")],
                  promote="Checkpoint/render/metric/config/evaluator hashes match.",fail="New cycle; never relabel mismatch.")
            continue
        com=f"P9-B04-{scene.upper()}-S0-COMMON-I5000"
        train(runs,com,"B04",scene,0,"common",0,5000,None,(b03,side[scene],evaluators[scene]),"Frozen seed0 scene checkpoint.","Exact provenance/budget under B03 freeze.","Diagnose once; no retune.",None,"gpu15")
        for mode in MODES:
            rid=f"P9-B04-{scene.upper()}-S0-{mode.upper()}-I6000"; seed0.append(rid)
            train(runs,rid,"B04",scene,0,mode,5001,6000,2048 if mode in {"capacity_only","full","shuffled"} else None,(com,side[scene],evaluators[scene]),
              "Frozen seed0 all-scene causal lane.","Terminal exact artifact and available metrics.","Record scene failure; no retune.",com,"gpu4")
    seed0dec=decide(runs,"P9-B04-S0-ANALYZE-DECISION","B04-seed0-decision","All-six seed0 passes exact seed-expansion predicate.",(*seed0,*evaluators.values()),
      "Annotated event conjunction, every static bound, no scene failure, control/shuffle pass.","No expansion; preserve seed0 evidence.","claim_grade")
    expanded=[]
    for seed in (1,2):
        for scene in SCENES:
            com=f"P9-B04-{scene.upper()}-S{seed}-COMMON-I5000"
            train(runs,com,"B04-seeds",scene,seed,"common",0,5000,None,(seed0dec,side[scene],evaluators[scene]),"Conditionally selected matched-seed checkpoint.","Provenance validates and seed differs.","Preserve selection conditionality.",None,"gpu15",True)
            for mode in MODES:
                rid=f"P9-B04-{scene.upper()}-S{seed}-{mode.upper()}-I6000"; expanded.append(rid)
                train(runs,rid,"B04-seeds",scene,seed,mode,5001,6000,2048 if mode in {"capacity_only","full","shuffled"} else None,(com,side[scene],evaluators[scene]),
                  "Conditionally selected matched-seed lane.","Terminal exact artifact and metrics.","Report failure; no tune.",com,"gpu4",True)
    decide(runs,"P9-B04-SEEDS-ANALYZE-DECISION","B04-seeds-decision","Selected seeds confirm/refute across six scenes.",expanded,
      "Report seed0 and conditional expanded estimands separately; apply frozen criteria.","Conclude partial/negative robustness.","claim_grade","cpu4",True)

    ids=[x["run_id"] for x in runs]; pos={v:i for i,v in enumerate(ids)}
    if len(ids)!=len(pos): raise RuntimeError("duplicate run ID")
    for x in runs:
        for d in x["prerequisites"]:
            if d not in pos: raise RuntimeError(f"{x['run_id']} unregistered prerequisite {d}")
            if pos[d]>=pos[x["run_id"]]: raise RuntimeError(f"{x['run_id']} cyclic prerequisite {d}")
            if not x["conditional"] and runs[pos[d]]["conditional"]:
                raise RuntimeError(f"{x['run_id']} unconditional entry depends ordinarily on conditional {d}")
        for group in x.get("conditional_prerequisite_sets", []):
            for d in group["required_run_ids"]:
                if d not in pos: raise RuntimeError(f"{x['run_id']} unregistered conditional prerequisite {d}")
                if pos[d]>=pos[x["run_id"]]: raise RuntimeError(f"{x['run_id']} cyclic conditional prerequisite {d}")
        t=x["training"]
        if t:
            if t["start_iteration"]==0 and t["input_checkpoint"] is not None: raise RuntimeError(f"{x['run_id']} common has input")
            if t["start_iteration"]>0 and t["input_checkpoint"] is None: raise RuntimeError(f"{x['run_id']} continuation lacks input")
            if t["input_checkpoint"] and t["input_checkpoint"]["path"]==t["output_checkpoint"]["path"]: raise RuntimeError(f"{x['run_id']} aliases checkpoints")
            if not x["configuration"]["derived_config_path"] or not x["configuration"]["derived_config_sha256"]:
                raise RuntimeError(f"{x['run_id']} lacks derived mode config")
            refs={(a["producer_run_id"],a["schema"]) for a in x["input_artifacts"] if "producer_run_id" in a}
            if t["mode"] in {"null_reset","capacity_only","visibility_only","full","shuffled"}:
                expected=(f"P9-P04-{x['scene'].upper()}-TRAIN-SIDECARS-FREEZE","phase9-train-sidecars-v1")
                if expected not in refs: raise RuntimeError(f"{x['run_id']} lacks train sidecar binding")
            if x["stage"] in {"B03","B04","B04-seeds"} and t["mode"] in {"null_reset","capacity_only","oracle_capacity","full","shuffled"}:
                if ("P9-B02-CAPACITY-ADMISSION-DECISION","phase9-optimizer-policy-v1") not in refs:
                    raise RuntimeError(f"{x['run_id']} lacks optimizer policy binding")
        if x["action"] in {"decide","score-gate-a"} and "predicate_binding" not in x:
            raise RuntimeError(f"{x['run_id']} lacks predicate binding")
    if sum(x["stage"]=="B03" for x in runs)!=7: raise RuntimeError("B03 needs seven lanes")
    null_run=next(x for x in runs if x["run_id"]=="P9-B03-CUT-S0-NULL_RESET-I6000")
    if null_run["training"]["requested_k"]!=2048: raise RuntimeError("B03 null-reset K must equal 2048")
    join=next(x for x in runs if x["run_id"]=="P9-B02-REPAIR-DECISION")
    if join["conditional"] or len(join.get("conditional_prerequisite_sets",[]))!=2 or len(join.get("conditional_input_artifacts",[]))!=2:
        raise RuntimeError("B02 repair join lacks unconditional typed two-branch resolution")
    for x in runs:
        if x["stage"]=="A06-baseline":
            if any("LABEL-FREEZE" in d for d in x["prerequisites"]): raise RuntimeError(f"{x['run_id']} depends on labels")
    for run_id in ("P9-A06-FLAME_STEAK-TRANSFER-SCORE","P9-A06-SEAR_STEAK-TRANSFER-SCORE"):
        x=runs[pos[run_id]]
        schemas={o["schema"] for o in x["expected_outputs"]}
        if not {"phase9-gate-a-score-v1","phase9-provenance-v1"}.issubset(schemas):
            raise RuntimeError(f"{run_id} lacks score/provenance outputs")
    for scene in SCENES:
        for st in ("P01","P02","P03","P04"):
            if not any(x["stage"]==st and x["scene"]==scene for x in runs): raise RuntimeError(f"missing {st} {scene}")
    paths={"objective":repo/"research-wiki/objectives/depth-visibility-capacity-v1.md",
      "method":repo/"research-wiki/operations/phase9-csvl-isr-v1-method.md",
      "slice_b_contract":repo/"research-wiki/operations/phase9-csvl-isr-v1-slice-b-contract.md",
      "plan":repo/"research-wiki/operations/phase9-csvl-isr-v1-experiment-plan.md",
      "split_manifest":repo/"configs/depth_visibility/n3v_split_v1.json",
      "annotation_windows":repo/"configs/depth_visibility/annotation_windows_v1.json",
      "scientific_config":repo/"configs/depth_visibility/csvl_isr_v1.json",
      "training_modes":repo/"configs/depth_visibility/isr_modes_v1.json",
      "generator":Path(__file__).resolve()}
    stage={}
    for x in runs: stage[x["stage"]]=stage.get(x["stage"],0)+x["max_gpu_hours"]
    un=sum(x["max_gpu_hours"] for x in runs if not x["conditional"]); con=sum(x["max_gpu_hours"] for x in runs if x["conditional"])
    return {"schema_version":"phase9-run-matrix-v2","cycle":"csvl-isr-v1","generated_before_new_phase9_outcomes":True,
      "base_commit_before_implementation":"94cd67df53cfc487989c71dc16a60fe853f53550","submission_ready":False,
      "submission_blocker":"P9-I01 must resolve every code/config/command/environment/artifact binding",
      "scene_order":list(SCENES),"annotated_scene_order":list(ANN),"initial_seed":0,"conditional_expansion_seeds":[1,2],
      "pilot_k":256,"comparable_k":2048,"common_iteration":5000,"pilot_iteration":5250,"comparable_iteration":6000,"point_ceiling":600000,
      "source_sha256":{k:sha(v) for k,v in paths.items()},"resource_authority":{"probe_date":"2026-07-15","partition":"boost_usr_prod",
        "gres":"gpu:a100:4","node_memory_mib":514000,"cpus":32,"partition_limit":"1-00:00:00","gpu_memory_gib":64,"gpu_memory_verify_at_I01":True},
      "run_count":len(runs),"unconditional_max_gpu_hours":un,"conditional_additional_max_gpu_hours":con,
      "registered_total_max_gpu_hours":un+con,"stage_max_gpu_hours":dict(sorted(stage.items())),
      "storage_maxima_gib_sum_not_concurrent":{"temporary":sum(x["storage"]["maximum_temporary_gib"] for x in runs),
        "durable_run":sum(x["storage"]["maximum_durable_run_gib"] for x in runs),
        "checkpoint_output":sum(x["storage"]["maximum_checkpoint_output_gib"] for x in runs)},"runs":runs}
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output",type=Path,default=Path("configs/depth_visibility/phase9_run_matrix_v1.json")); ap.add_argument("--check",action="store_true"); a=ap.parse_args()
    repo=Path(__file__).resolve().parents[1]; payload=build(repo); p=a.output if a.output.is_absolute() else repo/a.output
    text=json.dumps(payload,indent=2,sort_keys=True)+"\n"
    if a.check:
        if not p.exists() or p.read_text()!=text: raise SystemExit(f"stale or missing matrix: {p}")
        print(f"validated {p} ({payload['run_count']} entries)"); return
    p.parent.mkdir(parents=True,exist_ok=True); p.write_text(text)
    print(f"wrote {p} ({payload['run_count']} entries, {payload['registered_total_max_gpu_hours']:.1f} max GPU-hours)")
if __name__=="__main__": main()
