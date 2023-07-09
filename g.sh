#!/user/bin/env/ bash
nohup  python -u  new_main.py \
  --dataset='GDELT'\
  --relation-prediction \
  --space=r \
  --gpu=0 \
  --r_p \
  --re \
  --tk \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

