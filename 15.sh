#!/user/bin/env/ bash
nohup  python -u  new_main.py \
  --dataset='ICEWS05-15'\
  --relation-prediction \
  --space=r\
  --gpu=0 \
  --short \
  --long \
  --r_p \
  --lr=0.001 \
  --fuse=att \
  --r_fuse=short \
  --record \
  --model_record \
 &

