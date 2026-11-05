#!/usr/bin/env bash
# 4-camera tier on the REAL absence fixtures (2026-10-19): pull one finished run's held-out renders (frames 40-99;
# STG chunk = 50-99) from the work area, lay them out as ghost_ring_real.py expects (render/camXX/NNNN.png next to the
# fix/orig/mask packs already pulled for the monocular runs), and score ghost vs ring.
# Usage: real4c_score.sh METHOD SCENE RUN_ID      METHOD: multi4d | ex4dgs | 4c4d | stg4c
set -uo pipefail
M=$1; S=$2; RUN=$3
SSH=/c/Windows/System32/OpenSSH/ssh.exe; SCP=/c/Windows/System32/OpenSSH/scp.exe; H=siyengar@login.leonardo.cineca.it
R=/leonardo_work/EUHPC_D36_068/sri/proj_adags/fmd/runs; PK=/leonardo_scratch/large/userexternal/siyengar/fmd/pack
L=D:/adags-ars/fmd/real/r4c_$RUN; SRCPACK=D:/adags-ars/fmd/real/m4d_monoreal_${S}_rep1
mkdir -p $L/raw
case $M in
  multi4d) D=$($SSH -o BatchMode=yes $H "ls -d $R/multi4d/$RUN/final/iteration_*/render | tail -1"); FILES=$(for t in $(seq 40 99); do printf "%s.png " $t; done); CAMS=cam00; F0=40 ;;
  ex4dgs)  D=$R/ex4dgs/$RUN/model/test/itrs_40000/renders; FILES=$(for t in $(seq 40 99); do printf "%06d.png " $t; done); CAMS=cam00; F0=40 ;;
  4c4d)    D=$($SSH -o BatchMode=yes $H "ls -d $R/4c4d/$RUN/model_root/${RUN}_render/render/test/ours_*/renders | tail -1"); FILES=$(for k in 0 1 3; do for t in $(seq 40 99); do printf "%05d.png " $((k*300+t)); done; done); CAMS=cam00,cam05,cam17; F0=40 ;;
  stg4c)   D=$R/stg4c/$RUN/test/ours_25000/renders; FILES=$(for t in $(seq 0 49); do printf "%05d.png " $t; done); CAMS=cam00; F0=50 ;;
esac
$SSH -o BatchMode=yes $H "cd $D && tar cf $PK/r4c_$RUN.tar $FILES" || { echo "PACK FAILED $RUN"; exit 1; }
$SCP -q $H:$PK/r4c_$RUN.tar $L/raw/ && (cd $L/raw && tar xf r4c_$RUN.tar && rm r4c_$RUN.tar)
python -I - "$M" "$L" "$SRCPACK" "$CAMS" "$F0" <<'PY'
import os, sys, shutil
m, L, sp, cams, f0 = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4].split(","), int(sys.argv[5])
for c in cams:
    for sub in ("render", "fix", "orig", "mask"):
        os.makedirs(f"{L}/{sub}/{c}", exist_ok=True)
    k = {"cam00": 0, "cam05": 1, "cam17": 3}[c]
    for t in range(f0, 100):
        src = {"multi4d": f"{t}.png", "ex4dgs": f"{t:06d}.png", "4c4d": f"{k * 300 + t:05d}.png", "stg4c": f"{t - 50:05d}.png"}[m]
        shutil.copyfile(f"{L}/raw/{src}", f"{L}/render/{c}/{t:04d}.png")
        for sub in ("fix", "orig"):
            shutil.copyfile(f"{sp}/{sub}/{c}/{t:04d}.png", f"{L}/{sub}/{c}/{t:04d}.png")
    for t in range(60, 90):
        shutil.copyfile(f"{sp}/mask/{c}/{t:05d}.png", f"{L}/mask/{c}/{t:05d}.png")
print("laid out", cams)
PY
cd /d/adags-ars/fmd && python -I ghost_ring_real.py $L r4c_$RUN $CAMS $F0
