#!/user/bin/env/ bash
nohup  python -u  main.py \
  --dataset='ICEWS14s'\
  --relation-prediction \
  --gpu=1 \
  --long \
  --short \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

