import numpy as np

class CausalGridWorld:
    """
    7x7 网格。Agent 需要:
    1. 捡起钥匙 (K)
    2. 用钥匙开门 (D)
    3. 拿到宝藏 (T)
    
    因果结构: has_key=True → can_open_door=True → can_reach_treasure=True
    干扰变量: 墙上的装饰符号 (随机, 与任务无关) — 测试因果 vs 相关
    """
    ACTIONS = ['up', 'down', 'left', 'right', 'pickup', 'use']
    
    def __init__(self, size=7, seed=None):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        # 0=空, 1=墙, 2=钥匙, 3=门, 4=宝藏, 5=装饰(干扰)
        
        # 外墙
        self.grid[0,:] = 1; self.grid[-1,:] = 1
        self.grid[:,0] = 1; self.grid[:,-1] = 1
        # 分隔墙 (中间竖墙, 门在中间)
        self.grid[1:-1, self.size//2] = 1
        door_row = self.size // 2
        self.grid[door_row, self.size//2] = 3
        
        # 钥匙放左侧
        self.key_pos = (self.rng.randint(1, self.size-1), 
                        self.rng.randint(1, self.size//2))
        while self.grid[self.key_pos] != 0:
            self.key_pos = (self.rng.randint(1, self.size-1), 
                            self.rng.randint(1, self.size//2))
        self.grid[self.key_pos] = 2
        
        # 宝藏放右侧
        self.treasure_pos = (self.rng.randint(1, self.size-1), 
                             self.rng.randint(self.size//2+1, self.size-1))
        self.grid[self.treasure_pos] = 4
        
        # 干扰装饰 (关键: 与is_key_picked强相关但无因果)
        # 每当reset时, 若rng判定为"装饰日", 同时放装饰和让door更难开
        # 这是个虚假相关 — 好的agent应该学会忽略它
        for _ in range(3):
            r, c = self.rng.randint(1, self.size-1), self.rng.randint(1, self.size-1)
            if self.grid[r,c] == 0:
                self.grid[r,c] = 5
        
        self.agent_pos = [self.size//2, 1]
        self.has_key = False
        self.door_open = False
        self.done = False
        self.steps = 0
        self.max_steps = 100
        return self.get_obs()
    
    def get_obs(self):
        """部分可观测: 3x3 视野 + 内部状态"""
        r, c = self.agent_pos
        view = np.zeros((3, 3), dtype=np.int32)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    view[dr+1, dc+1] = self.grid[nr, nc]
                else:
                    view[dr+1, dc+1] = 1  # 边界当墙
        
        # One-hot 视野 (6类 × 9格 = 54) + 内部状态(2) = 56
        view_oh = np.zeros((3, 3, 6), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                view_oh[i, j, view[i,j]] = 1.0
        
        obs = np.concatenate([
            view_oh.flatten(),
            [float(self.has_key), float(self.door_open)]
        ])
        return obs.astype(np.float32)  # shape (56,)
    
    def step(self, action):
        self.steps += 1
        reward = -0.01  # 步长惩罚
        
        r, c = self.agent_pos
        if action == 0 and r > 0: new_pos = [r-1, c]
        elif action == 1 and r < self.size-1: new_pos = [r+1, c]
        elif action == 2 and c > 0: new_pos = [r, c-1]
        elif action == 3 and c < self.size-1: new_pos = [r, c+1]
        else: new_pos = [r, c]
        
        # 移动检查
        if action < 4:
            cell = self.grid[new_pos[0], new_pos[1]]
            if cell == 1:  # 墙
                pass
            elif cell == 3 and not self.door_open:  # 关着的门
                pass
            else:
                self.agent_pos = new_pos
        
        elif action == 4:  # pickup
            if self.grid[r, c] == 2:  # 在钥匙上
                self.has_key = True
                self.grid[r, c] = 0
                reward += 0.1  # 小奖励鼓励进度
        
        elif action == 5:  # use
            # 检查相邻是否有门
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.size and 0<=nc<self.size:
                    if self.grid[nr,nc] == 3 and self.has_key:
                        self.door_open = True
                        reward += 0.2
        
        # 检查宝藏
        if self.agent_pos[0] == self.treasure_pos[0] and \
           self.agent_pos[1] == self.treasure_pos[1]:
            reward += 1.0
            self.done = True
        
        if self.steps >= self.max_steps:
            self.done = True
        
        return self.get_obs(), reward, self.done, {}
    
    def intervene(self, variable, value):
        """干预接口 — 用于反事实训练"""
        if variable == 'has_key':
            self.has_key = value
        elif variable == 'door_open':
            self.door_open = value
        return self.get_obs()