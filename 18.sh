#!/user/bin/env/ bash
nohup  python -u  new_main.py \
  --dataset='ICEWS18'\
  --relation-prediction \
  --space=r \
  --gpu=2 \
  --re \
  --tk \
  --r_p \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

