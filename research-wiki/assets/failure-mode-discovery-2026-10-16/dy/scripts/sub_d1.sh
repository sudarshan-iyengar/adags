# usage: bash sub_d1.sh REPS   (e.g. "1" or "2 3")
source /leonardo_work/EUHPC_D36_068/sri/proj_adags/agent-control/ars/ars_sub.sh; X=/leonardo_scratch/large/userexternal/siyengar/fmd
V="v1:DY_MODE=off v2:DY_MODE=window,DY_R=8 v3:DY_MODE=window,DY_R=32 v4:DY_MODE=soft v5:DY_MODE=reach v6:DY_MODE=shuffle"
for r in $REPS; do for s in apple paper-windmill spin; do for vv in $V; do
  v=${vv%%:*}; envs=${vv#*:}; [ "$v" = v6 ] && envs="$envs,DY_SEED=$r"
  GO=--go sub fmd_dy_d1_${s}_${v}_r$r FMD-DY-D1 "" "fmd dy Pilot 1 (D1 fusion reach): $s variant $v ($envs) rerun $r" --export=ALL,SCENE=$s,RUN_ID=d1_${s}_${v}_r$r,REPO=$X/repo/mosca_d1,$envs $X/agent/dy/fmd_dy_mosca.sbatch
done; done; done 2>&1 | grep -- "->"
