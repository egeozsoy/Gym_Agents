- Pytorch Deep Q Learning for openai gym enviroments
- Mostly used reward maximization. Which means the loss is -1 * expected reward. To minimize the loss, pytorch has to maximize the expected reward.
- I used different resources for coming up with solutions and mention them in my code. Not everything in this repo is my idea.



TRAIN
 python -m baselines.run --alg=deepq --env=CartPole-v1 --num_timesteps=200000 --network=mlp --save_path=models/cartpole_deepq
 
VISUALIZE
python -m baselines.run --alg=deepq --env=CartPole-v1 --num_timesteps=0 --load_path=models/cartpole_deepq --play

