#!/bin/bash
# fmd mono2: download DyCheck iPhone as processed by Shape of Motion (Drive folder from its README) and by MoSca (Drive zip from its README).
X=/leonardo_scratch/large/userexternal/siyengar/fmd; D=$X/data/dycheck; mkdir -p $D/som $D/mosca_zip $X/logs
source $X/envs/som/bin/activate
exec > $X/logs/dl_dycheck.log 2>&1
echo "start $(date -Is)"
( cd $D/mosca_zip && gdown 1sSvVi5Bid_KQsuguVGuqUVgWPzVdb9jM -O mosca_data.zip && echo "MOSCA ZIP DONE $(date -Is)" && ls -la ) &
( cd $D/som && gdown --folder 1xJaFS_3027crk7u36cue7BseAX80abRe --remaining-ok && echo "SOM FOLDER DONE $(date -Is)" && ls -laR | head -50 ) &
wait; echo "all done $(date -Is)"
