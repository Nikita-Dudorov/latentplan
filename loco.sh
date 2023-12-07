#!/bin/bash
export PYTHONPATH=.:$PYTHONPATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/nikita/.mujoco/mujoco210/bin

name=T-1
datasets=(hopper-medium-replay-v2)  # hopper-medium-v2 hopper-medium-expert-v2)
device=cpu

for round in {1..1}; do
  for data in ${datasets[@]}; do
    python3 scripts/train.py --dataset $data --exp_name $name-$round --tag development --seed $round --device $device
    python3 scripts/trainprior.py --dataset $data --exp_name $name-$round --device $device
    for i in {1..20}; do
      python3 scripts/plan.py --test_planner beam_prior --dataset $data --exp_name $name-$round --suffix $i --beam_width 64 --n_expand 4 --horizon 15 --device $device
    done
  done
done

for data in ${datasets[@]}; do
  for round in {1..1}; do
    python3 plotting/read_results.py --exp_name $name-$round --dataset $data
  done
done


