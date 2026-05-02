import numpy as np

class CausalGridWorld:
    ACTIONS = ['up', 'down', 'left', 'right', 'pickup', 'use']
    
    def __init__(self, size=7, seed=None):
        self.size = size
        self.rng = np.random.RandomState(seed)
        self.reset()
    
    def reset(self):
        self.grid = np.zeros((self.size, self.size), dtype=np.int32)
        self.grid[0,:] = 1; self.grid[-1,:] = 1
        self.grid[:,0] = 1; self.grid[:,-1] = 1
        self.grid[1:-1, self.size//2] = 1
        door_row = self.size // 2
        self.grid[door_row, self.size//2] = 3
        
        self.key_pos = self._place_item(2, 1, self.size//2-1)
        self.treasure_pos = self._place_item(4, self.size//2+1, self.size-1)
        
        for _ in range(3):
            r, c = self.rng.randint(1, self.size-1), self.rng.randint(1, self.size-1)
            if self.grid[r,c] == 0:
                self.grid[r,c] = 5
        
        self.agent_pos = [self.size//2, 1]
        self.has_key = False
        self.door_open = False
        self.done = False
        self.steps = 0
        self.max_steps = 80
        self.prev_key_dist = self._dist_to(self.key_pos)
        self.prev_door_dist = self._dist_to((self.size//2, self.size//2))
        self.prev_treasure_dist = self._dist_to(self.treasure_pos)
        self._picked_key = False
        self._opened_door = False
        return self.get_obs(), self.get_ground_truth()
    
    def _place_item(self, item, c_min, c_max):
        for _ in range(50):
            r = self.rng.randint(1, self.size-1)
            c = self.rng.randint(c_min, c_max+1)
            if self.grid[r,c] == 0:
                self.grid[r,c] = item
                return (r, c)
        for r in range(1, self.size-1):
            for c in range(c_min, c_max+1):
                if self.grid[r,c] == 0:
                    self.grid[r,c] = item
                    return (r, c)
        return (1, c_min)
    
    def _dist_to(self, pos):
        return abs(self.agent_pos[0]-pos[0]) + abs(self.agent_pos[1]-pos[1])
    
    def get_ground_truth(self):
        """返回4个因果变量的真实值: [has_key, door_open, near_treasure, saw_decoration]"""
        near_treasure = 1.0 if self._dist_to(self.treasure_pos) <= 2 else 0.0
        # 检查视野内是否有装饰
        r, c = self.agent_pos
        saw_deco = 0.0
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r+dr, c+dc
                if 0<=nr<self.size and 0<=nc<self.size and self.grid[nr,nc]==5:
                    saw_deco = 1.0
        return np.array([float(self.has_key), float(self.door_open), 
                         near_treasure, saw_deco], dtype=np.float32)
    
    def get_obs(self):
        r, c = self.agent_pos
        view = np.zeros((3, 3), dtype=np.int32)
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                nr, nc = r+dr, c+dc
                if 0 <= nr < self.size and 0 <= nc < self.size:
                    view[dr+1, dc+1] = self.grid[nr, nc]
                else:
                    view[dr+1, dc+1] = 1
        view_oh = np.zeros((3, 3, 6), dtype=np.float32)
        for i in range(3):
            for j in range(3):
                view_oh[i, j, view[i,j]] = 1.0
        obs = np.concatenate([view_oh.flatten(), [float(self.has_key), float(self.door_open)]])
        return obs.astype(np.float32)
    
    def step(self, action):
        self.steps += 1
        reward = 0.0
        
        r, c = self.agent_pos
        if action == 0 and r > 0: new_pos = [r-1, c]
        elif action == 1 and r < self.size-1: new_pos = [r+1, c]
        elif action == 2 and c > 0: new_pos = [r, c-1]
        elif action == 3 and c < self.size-1: new_pos = [r, c+1]
        else: new_pos = [r, c]
        
        if action < 4:
            cell = self.grid[new_pos[0], new_pos[1]]
            if cell == 1: new_pos = [r, c]
            elif cell == 3 and not self.door_open: new_pos = [r, c]
            else: self.agent_pos = new_pos
        elif action == 4:
            if self.grid[r, c] == 2 and not self.has_key:
                self.has_key = True
                self.grid[r, c] = 0
                reward += 0.5
                self._picked_key = True
        elif action == 5:
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<self.size and 0<=nc<self.size:
                    if self.grid[nr,nc] == 3 and self.has_key and not self.door_open:
                        self.door_open = True
                        reward += 0.5
                        self._opened_door = True
        
        # 塑形奖励
        if not self._picked_key:
            new_dist = self._dist_to(self.key_pos)
            reward += (self.prev_key_dist - new_dist) * 0.1
            self.prev_key_dist = new_dist
        if self._picked_key and not self._opened_door:
            door_pos = (self.size//2, self.size//2)
            new_dist = self._dist_to(door_pos)
            reward += (self.prev_door_dist - new_dist) * 0.1
            self.prev_door_dist = new_dist
        if self._opened_door:
            new_dist = self._dist_to(self.treasure_pos)
            reward += (self.prev_treasure_dist - new_dist) * 0.1
            self.prev_treasure_dist = new_dist
        
        reward -= 0.01
        
        if self.agent_pos[0] == self.treasure_pos[0] and \
           self.agent_pos[1] == self.treasure_pos[1]:
            reward += 1.0
            self.done = True
        if self.steps >= self.max_steps:
            self.done = True
        
        return self.get_obs(), reward, self.done, self.get_ground_truth()