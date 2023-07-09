#!/user/bin/env/ bash
nohup  python -u  new_main.py \
  --dataset='ICEWS14s'\
  --relation-prediction \
  --space=r\
  --gpu=1 \
  --r_p \
  --long \
  --short \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

