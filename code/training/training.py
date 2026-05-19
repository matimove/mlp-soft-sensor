from algorithms.q_learning import Qlearn
from  algorithms.neural_net import NN
from game.yatzy import Yatzy
from algorithms.replaybuffer import Buffer

#---------------------#
agent = NN()
agent.initialize()
#---------------------#
env = Yatzy()
#---------------------#
# How many games to train the agent on
episodes = 100_000
#---------------------#
# Determine size of replay buffer and batch size that is sampleded during each turn
buffer = Buffer(capacity=100_000)
batch_size =  8
#---------------------#
scores = []
bonuses = []

for episode in range(episodes):
    
    state = env.reset()

    done = False

    turns = 0

    while True:

        turns += 1

        action = agent.choose_action(state)

        next_state, reward, done, final_score, info = env.step(action)
        
        buffer.push(state, action, reward, next_state)

        if len(buffer) >= 5000:
            batch = buffer.sample(batch_size)
            agent.update_batch(batch)

        state = next_state

        if done:
            scores.append(final_score[0])
            bonuses.append(final_score[1])
            break

    agent.decay_epsilon()
    
    if episode % 100 == 0:
        avg_score = sum(scores[-100:]) / len(scores[-100:])
        avg_bonus = sum(bonuses[-100:]) / len(bonuses[-100:]) * 100
        print(f"Episode {episode} | Avg Score: {avg_score:.2f} | Bonus %: {avg_bonus:.2f} | Epsilon: {agent.epsilon:.3f}")
        

