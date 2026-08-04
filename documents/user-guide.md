User-guide

Simple instructions:

git clone "repo-address"
cd "repo-name"
pip install -r requirements.txt (consider starting a virtual environment first)
python -m training.training

----------------------------------------------------------------

Run the training.py script to begin training. (I ran it from the code folder using "python -m training.training")
Then the training loop should start and you will begin to see updates:

Episode 0 | Avg Score: 50.00 | Bonus %: 0.00 | Epsilon: 0.999
Episode 100 | Avg Score: 50.11 | Bonus %: 0.00 | Epsilon: 0.904
...

Some important parameters that can be changed include

training.py: Changing number of games played and buffersize and batchsize

algorithms/neural_net.py: Changing the learning parameters in the __init__() function.

yatzy.py: Changing the rewards given in the generate_reward() function

----------------------------------------------------------------



