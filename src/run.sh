#!/bin/bash

n_trial=1
env=MountainCarContinuous-v0
batchnorm=false
is_training=true
alpha=0.6
beta0=0.9
gamma=0.75
tmax=200

for trial_i in $(seq "$n_trial")
do
    echo "running ICNN $trial_i"
    python3 main.py --model ICNN --env $env --outdir output --total_episode 100 --train_episode 5 \
                    --test_episode 5 --trial_i $trial_i --outheta 0.15 --ousigma 0.2 --icnn_bn $batchnorm \
                    --alpha $alpha --beta0 $beta0 --force true --is_training $is_training \
                    --tmax $tmax --discount $gamma
done
python3 archive/plot-single.py --model_path output/ICNN/ --model ICNN --n_trial $n_trial


for trial_i in $(seq "$n_trial")
do
    echo "running DDPG $trial_i"
    python3 main.py --model DDPG --env $env --outdir output --total_episode 100 --train_episode 5 \
                    --test_episode 5 --outheta 0.15 --ousigma 0.2 \
                    --alpha $alpha --beta0 $beta0 --force true --trial_i $trial_i --is_training $is_training \
                    --tmax $tmax --discount $gamma
done

python3 archive/plot-single.py --model_path output/DDPG/ --model DDPG --n_trial $n_trial


for trial_i in $(seq "$n_trial")
do
    echo "running NAF $trial_i"
    python3 main.py --model NAF --env $env --outdir output --total_episode 100 --train_episode 5 \
                    --test_episode 5 --outheta 0.15 --ousigma 0.2 --naf_bn $batchnorm \
                    --alpha $alpha --beta0 $beta0 --force true --trial_i $trial_i --is_training $is_training \
                    --tmax $tmax --discount $gamma
done

python3 archive/plot-single.py --model_path output/NAF/ --model NAF --n_trial $n_trial

