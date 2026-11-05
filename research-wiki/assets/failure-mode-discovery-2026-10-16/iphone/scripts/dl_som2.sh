#!/bin/bash
# fmd mono2: Shape of Motion processed iPhone scenes (README Drive folder 1xJaFS_3027crk7u36cue7BseAX80abRe), per-file, parallel.
X=/leonardo_scratch/large/userexternal/siyengar/fmd; D=$X/data/dycheck/som/iphone; cd $D; source $X/envs/som/bin/activate
get() { [ -f $1.zip ] && { echo "have $1"; return; }; gdown -q $2 -O $1.zip && echo "GOT $1 $(date -Is)" || echo "FAILED $1"; }
unz() { mkdir -p ../scenes && unzip -q -o $1.zip -d ../scenes/ && echo "UNZIPPED $1 $(date -Is)" && rm -f $1.zip; }
( unz apple ) &
( get block 1b9Y-hUm9Cviuq-fl7gG-q7rUK7j0u2Rv; unz block ) &
( get paper-windmill 15PirJRqsT5lLjuGdLWALBDFMQanj8FTh; unz paper-windmill ) &
( get spin 1055wcQk-ZfVWXa_g-dpQIRQy-kLBL_Lk; unz spin ) &
( get teddy 1Mqm4C1Oitv4AsDM2n0Ojbt5pmF_qXVfI; unz teddy ) &
wait; echo "CORE DONE $(date -Is)"
( unz backpack ) &
( get creeper 1inkHp24an1TyWvBekBxu2wRIyLQ0gkhO; unz creeper; get handwavy 1frv8miU24Dl7fqblYt7zkwj129ci-68U; unz handwavy ) &
( get haru-sit 1BmuxJXKi6dVaNOjmppuETQsaspAV9Wca; unz haru-sit; get mochi-high-five 1OpgF2ILf43jcN-226wQcxImjcfMAVOwA; unz mochi-high-five ) &
( get pillow 1Uc2BXpONnWhxKNs6tKMle0MiSVMVZsuB; unz pillow; get sriracha-tree 18sjQQMU6AijyXg4BoucLX82R959BYAzz; unz sriracha-tree ) &
wait; echo "ALL DONE $(date -Is)"
