#!/user/bin/env/ bash
nohup  python -u  main.py \
  --dataset='GDELT'\
  --relation-prediction \
  --gpu=0 \
  --short \
  --long \
  --lr=0.001 \
  --fuse=gate \
  --r_fuse=gate \
  --record \
  --model_record \
 &

