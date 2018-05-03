#!/bin/bash

n_trial=5
env=MountainCarContinuous-v0

python3 main.py --model ICNN --env $env --outdir output --total_episode 1000 --train_episode 100 \
                --test_episode 5 --n_trial $n_trial --outheta 0.15 --ousigma 0.2 

python3 main.py --model DDPG --env $env --outdir output --total_episode 1000 --train_episode 100 \
                --test_episode 5 --n_trial $n_trial --outheta 0.15 --ousigma 0.1 

python3 main.py --model NAF --env $env --outdir output --total_episode 1000 --train_episode 100 \
                --test_episode 5 --n_trial $n_trial --outheta 0.15 --ousigma 0.1 
