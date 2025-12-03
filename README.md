# RL Project

# Requirements

## Hyperparameter optimization
### Start by running this command
```sh
python -m rl_zoo3.train --algo sac --env PulleyEnv-v0 --gym-packages env --env-kwargs dt:0.001 max_steps:2500 -conf hyperparams\pulley.yaml -n 1000000 -optimize --n-trials 2 --n-jobs 2 --sampler tpe --pruner median --study-name pulley_sac_hpo --storage sqlite:///pulley_sac_hpo.db --verbose 1 --vec-env dummy --num-threads 1 --eval-episodes 300
```
### Command to pull up stats for study
```sh
python create_yaml.py
```
### Run this command after to create yaml file to be used for training later on
```sh
python create_yaml.py
```

## Training
### Command to run training for the agent
```sh
python -m rl_zoo3.train --algo sac --env PulleyEnv-v0 --gym-packages env --env-kwargs dt:0.001 max_steps:2500 -conf hyperparams/pulley_best.yaml -n 20000000 --tensorboard-log tb --verbose 2 --log-interval 1000 --eval-freq 500000 --eval-episodes 500 --vec-env dummy --num-threads 1 
```

## Command to bring up tensorboard view training stats
```sh
tensorboard --logdir tb --port 6006
```

## Model Details
Instead of predicting tau1 and tau2, the model predicts delta which determines the resultant torque.

3 types of models were tested, one which predicts tau1 and tau2 to achieve a desired coactivation force and positon

The second model parameterizes tau1 and tau2 using the following control structure

The third model uses the same control structure as before but instead of predicting the base torque and delta it only predicts delta