import yaml

class RewardConfig:
    def __init__(self, path='configs/config.yaml'):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        self.cfg = cfg['reward']

    def get(self, key):
        return self.cfg.get(key, 0)

def calc_reward(prev, curr, events, reward_cfg):
    reward = 0.0
    if events.get('done'):
        reward += reward_cfg.get('finish_game')
    if events.get('score_increased'):
        reward += reward_cfg.get('shoot_enemy') * events.get('score_delta', 0)
    if events.get('enemy_removed'):
        reward += reward_cfg.get('remove_enemy') * events.get('enemy_delta', 0)
    if events.get('life_lost'):
        reward += reward_cfg.get('lose_life') * events.get('life_delta', 0)
    if events.get('survived'):
        reward += reward_cfg.get('survive_frame')
    if events.get('avoided_enemy'):
        reward += reward_cfg.get('avoid_enemy')
    if events.get('fast_kill'):
        reward += reward_cfg.get('fast_kill_bonus')
    if events.get('idle'):
        reward += reward_cfg.get('idle_penalty')
    if events.get('at_edge'):
        reward += reward_cfg.get('edge_penalty')
    if events.get('round_advanced'):
        reward += reward_cfg.get('round_bonus') * events.get('round_delta', 1)
    if events.get('cleared_wave'):
        reward += reward_cfg.get('cleared_wave')
    if events.get('missed_shots'):
        reward += reward_cfg.get('missed_shot_penalty') * events.get('missed_shots')
    if events.get('formation_broken'):
        reward += reward_cfg.get('formation_broken')
    if events.get('row_broken'):
        reward += reward_cfg.get('row_broken')
    if events.get('column_broken'):
        reward += reward_cfg.get('column_broken')
    if events.get('cluster_broken'):
        reward += reward_cfg.get('cluster_broken')
    if events.get('too_close_to_enemy'):
        reward += reward_cfg.get('too_close_to_enemy')
    if events.get('safe_distance'):
        reward += reward_cfg.get('safe_distance')
    if events.get('very_safe_distance'):
        reward += reward_cfg.get('very_safe_distance')
    return reward 