am221 project on ICNN

- project todos on trello: https://trello.com/b/xnKmZ4XY


## getting started
```bash
# to run
cd /src/icnn/src
python main.py --model ICNN --env MountainCarContinuous-v0 --outdir output \
  --total 100000 --train 100 --test 1 --tfseed 0 --npseed 0 --gymseed 0

# to monitor
tensorboard --logdir=output/board/
```

## Acknowledgements

Project | License
---|---|
| [locuslab/icnn](https://github.com/locuslab/icnn) | Apache 2.0 |
| [SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg) | MIT |
