#!/user/bin/env/ bash
nohup  python -u  main.py \
  --dataset='ICEWS18'\
  --relation-prediction \
  --gpu=2 \
  --short \
  --long \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

