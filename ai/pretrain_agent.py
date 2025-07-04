import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import yaml
from ai.agent import get_best_agent, pretrain_agent
from ai.simulator import SimpleSpaceInvadersSim, ACTIONS
import numpy as np

if __name__ == '__main__':
    with open('configs/config.yaml') as f:
        config = yaml.safe_load(f)
    agent_cfg = config.get('agent', {})
    input_shape = (3, 84, 84)
    n_actions = 5
    agent = get_best_agent(input_shape, n_actions, device='cuda', config=agent_cfg)

    # --- Zaawansowany pretraining z logami ---
    sim = SimpleSpaceInvadersSim()
    n_steps = 10000000
    batch_size = 64
    steps = 0
    episode = 0
    percent_last = 0
    while steps < n_steps:
        state = sim.reset()
        done = False
        ep_reward = 0
        ep_steps = 0
        while not done:
            action = np.random.randint(0, n_actions)
            next_state, reward, done = sim.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            ep_reward += reward
            ep_steps += 1
            steps += 1
            # Loguj co 1% postępu
            percent = int(100 * steps / n_steps)
            if percent > percent_last:
                print(f'Pretraining: {percent}% ({steps}/{n_steps})')
                percent_last = percent
            if steps % batch_size == 0:
                agent.update()
            if steps >= n_steps:
                break
        episode += 1
        print(f'Epizod {episode}: reward={ep_reward:.1f}, steps={ep_steps}')
    agent.save()
    print('Model zapisany po pretreningu.') 