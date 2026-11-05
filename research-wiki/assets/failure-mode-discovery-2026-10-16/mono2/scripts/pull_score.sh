#!/bin/bash
# fmd mono2: pull held-out renders (frames 40-99, cam00/05/17) of a Shape of Motion or MoSca run, build a scorer pack
# (render/ + the fix/orig/mask of the Multi4D monocular pack of the same scene), run ghost_ring_real.py and whole-frame PSNR.
# Usage: pull_score.sh LABEL SCENE REMOTE_RENDER_DIR KIND   (KIND = som: camXX/NNNN.png ; mosca: {1,2,3}_NNNNN.png)
set -euo pipefail
L=$1; SC=$2; RD=$3; K=$4
SSH="/c/Windows/System32/OpenSSH/ssh.exe -o BatchMode=yes siyengar@login02-ext.leonardo.cineca.it"
D=/d/adags-ars/fmd/real/$L; REF=/d/adags-ars/fmd/real/m4d_monoreal_${SC}_rep1
rm -rf $D; mkdir -p $D
$SSH "cd $RD && python3 -c \"
import os,tarfile,sys
k='$K'; t=tarfile.open(fileobj=sys.stdout.buffer,mode='w|')
for i,c in [(1,'cam00'),(2,'cam05'),(3,'cam17')]:
  for f in range(40,100):
    src=f'{c}/{f:04d}.png' if k=='som' else f'{i}_{f:05d}.png'
    t.add(src,arcname=f'render/{c}/{f:04d}.png')
t.close()\"" | tar xf - -C $D
for x in fix orig mask; do cp -r $REF/$x $D/; done
cd /d/adags-ars/fmd && python ghost_ring_real.py $D $L
python - $D $L <<'PY'
import sys, json, numpy as np
from PIL import Image
d, L = sys.argv[1:3]; out = {}
for c in ["cam00","cam05","cam17"]:
    p = []
    for t in range(40,100):
        r = np.asarray(Image.open(f"{d}/render/{c}/{t:04d}.png").convert("RGB")).astype(np.float32)/255
        f = np.asarray(Image.open(f"{d}/fix/{c}/{t:04d}.png").convert("RGB")).astype(np.float32)/255
        p.append(-10*np.log10(((r-f)**2).mean()))
    out[c] = {"wholeframe_psnr_40_99": round(float(np.mean(p)),2), "pre_gap_40_57": round(float(np.mean(p[:18])),2), "gap_60_89": round(float(np.mean(p[20:50])),2)}
    print(L, c, out[c])
json.dump(out, open(f"D:/adags-ars/fmd/scores/wholeframe_{L}.json","w"), indent=1)
PY
