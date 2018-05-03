n_trial = 5
env = MountainCarContinuous-v0
python3 main.py --model DDPG --env "$env" --outdir output --total 20000 --train 3000 --test 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"
python3 main.py --model NAF --env "$env" --outdir output --total 20000 --train 3000 --test 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"
python3 main.py --model iCNN --env "$env" --outdir output --total 20000 --train 3000 --test 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"
