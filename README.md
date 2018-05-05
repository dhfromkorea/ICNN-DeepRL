# Application of Input Convex Neural Network
- an AM221 project

## getting started
```bash
# to run experiments
cd src/
chmod +x run.sh
./run.sh

# to monitor
tensorboard --logdir=output/board/
```

## experiments

### MountainCarContinuous-v0

![negative q][src/figures/neg_q.png]
![simple neural network with discretized action space][src/figures/reward_DNN.png]
![icnn][src/figures/reward_ICNN.png]
![naf][src/figures/reward_NAF.png]
![ddpg][src/figures/reward_DDPG.png]


## Acknowledgements

Project | License
---|---|
| [locuslab/icnn](https://github.com/locuslab/icnn) | Apache 2.0 |
| [SimonRamstedt/ddpg](https://github.com/SimonRamstedt/ddpg) | MIT |
| [openai/baselines](https://github.com/openai/baselines) | MIT |
