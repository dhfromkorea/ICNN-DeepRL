n_trial = 5
env = MountainCarContinuous-v0
python3 main.py --model ICNN --env "$env" --outdir output --total_episode 1000 --train_episode 100 --test_episode 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"

python3 main.py --model DDPG --env "$env" --outdir output --total_episode 1000 --train_episode 100 --test_episode 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"

python3 main.py --model NAF --env "$env" --outdir output --total_episode 1000 --train_episode 100 --test_episode 5 --tfseed 0 --npseed 0 --gymseed 0 --n_trial "$n_trial"
