#!/user/bin/env/ bash
nohup  python -u  main.py \
  --dataset='ICEWS05-15'\
  --relation-prediction \
  --gpu=0 \
  --short \
  --long \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=short \
  --record \
  --model_record \
 &

