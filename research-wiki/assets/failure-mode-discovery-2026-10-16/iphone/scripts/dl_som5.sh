#!/bin/bash
# teddy: retry the Drive download (no login-node unzip), then submit its SoM job with ZIPFILE (the job unzips on the compute node)
X=/leonardo_scratch/large/userexternal/siyengar/fmd; D=$X/data/dycheck/som/iphone; cd $D; source $X/envs/som/bin/activate
for k in 1 2 3 4 5 6 7 8; do
  rm -f teddy.zip*.part
  if gdown -q 1Mqm4C1Oitv4AsDM2n0Ojbt5pmF_qXVfI -O teddy.zip; then
    echo "GOT teddy try $k $(date -Is)"
    source /leonardo_work/EUHPC_D36_068/sri/proj_adags/agent-control/ars/ars_sub.sh
    GO=--go sub fmd_mono2_somiph_teddy FMD-MONO2 "" "fmd mono2: Shape of Motion on untouched DyCheck iPhone teddy (job unzips the release zip first) + evaluate_iphone.py" --time=08:00:00 --export=ALL,SCENE=teddy,ZIPFILE=$D/teddy.zip $X/agent/fmd_mono2_som_iphone.sbatch
    exit 0
  fi
  echo "teddy try $k failed $(date -Is)"; sleep 1200
done
