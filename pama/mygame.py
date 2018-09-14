import numpy as np 
import sys
from numpy import random as random

class Pacman():
   
    def __init__(self, dir='smallGrid.txt'):
        # UP - DOWN - LEFT - RIGHT - STOP
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        self.load_map(dir)
        self.reset()


    def load_map(self, dir):
        f = open(dir, 'r')
        self.height, self.width = [int(x) for x in next(f).split()]
        self.state_ = np.zeros([4, self.height, self.width], dtype=np.float32)
         # state: wall -> dot -> pac -> ghosts
        self.total_dot = 0
        for x in range(self.height):
            line = next(f)
            for y in range(len(line)):
                if line[y] == '#':
                    self.state_[0,:,:][x, y] = 1
                elif line[y] == '.':
                    self.state_[1,:,:][x][y] = 1
                    self.total_dot += 1

        next(f)
        #f.readline()

        self.pac_ = [int(x)-1 for x in next(f).split()]
        self.state_[2,:,:][self.pac_[0], self.pac_[1]] = 1
        
        self.ghosts_ = []
        self.direction = []
        for line in f:
            x, y = [int(z)-1 for z in line.split()]
            self.state_[3,:,:][x, y] = 1
            self.ghosts_.append([x, y])  
            self.direction.append(0)

        f.close()
       
      
    def reset(self, nb_bigdot=2):
        self.state = np.copy(self.state_)
        self.pac = np.copy(self.pac_)
        self.ghosts = np.copy(self.ghosts_)
        # print(self.ghosts)
        # random bigdot

        self.my_dot    = 0

        if nb_bigdot == 4:
            x = [random.randint(1, int(self.height/3)),  random.randint(1, int(self.height/3)), random.randint(int(self.height/1.4), self.height-2), random.randint(int(self.height/1.4), self.height-2)]
            while nb_bigdot > 0:
                if nb_bigdot % 2 == 0:
                    y = random.randint(1, int(self.width/3.5))
                else:
                    y = random.randint(int(self.width/1.5), self.width-2)
                if self.state[1,:,:][x[nb_bigdot-1], y] == 1 and np.sum(self.get_legal_action([x[nb_bigdot-1], y])) == 2:
                    self.state[1,:,:][x[nb_bigdot-1], y] = 10
                    nb_bigdot -= 1
        elif nb_bigdot == 2:
            self.state[1,:,:][1, 1] = 10
            self.state[1,:,:][self.height-2, self.width-2] = 10


    def get_action_size(self):
        return len(self.actions)

    def get_state_size(self):
        return np.asarray(self.state.shape)

    def get_state(self):
        return self.state, self.get_legal_action(self.pac)

    def get_total_dot(self):
        return self.total_dot

    def get_legal_action(self, pos, isFinished=False):
        legal_action = [0, 0, 0, 0]
        if not isFinished:
            for i in range(len(self.actions)):
                newpos = self.move(pos, self.actions[i])
                if self.state[0,:,:][newpos[0], newpos[1]] == 0:
                    legal_action[i] = 1
        return np.array(legal_action)

    def move(self, pos, action):
        pos = np.add(pos, action)
        pos[0] = (pos[0] + self.height)%self.height
        pos[1] = (pos[1] + self.width)%self.width
        return pos

    def ghost_move(self):
        for i in range(len(self.ghosts)):
            pos = self.ghosts[i]
            legal_action = self.get_legal_action(pos)
            if np.sum(legal_action) >= 3 and legal_action[0] == 1 and pos[1] > self.width/2.5 and pos[1] < self.width/1.6:
                self.direction[i] = 0 
            else: 
                if not (legal_action[self.direction[i]] == 1 and (np.sum(legal_action) <= 2 or random.random() > 0.8)):
                    list_action,  = np.where(legal_action == 1)
                    index = random.randint(0, len(list_action)-1)
                    self.direction[i] = list_action[index]

            pos1 = self.move(pos, self.actions[self.direction[i]])
            self.state[3,:,:][pos[0], pos[1]] -= 1
            self.ghosts[i] = pos1
            self.state[3,:,:][pos1[0], pos1[1]] += 1

    def select_action(self):
        legal_action = self.get_legal_action(self.pac)
        actions = [0,1,2,3]
        # actions = [up, down, right, left]
        n = np.sum(legal_action)
        id = random.randint(1, n+1)
        dd = 0
        ida = 0
        for i in range(4):
            if (legal_action[i] == 1):
                dd += 1
                if dd == id:
                    ida = i
                    break
        return actions[ida]

    def gameover(self):
        if self.state[3,:,:][self.pac[0], self.pac[1]] == 1:
            return True
        if self.state[0,:,:][self.pac[0], self.pac[1]] == 1:
            return True
        return False

    def make_action(self, action):
        index = action
        pac_action = self.actions[index]
        self.state[2,:,:][self.pac[0], self.pac[1]] = 0
        self.pac = self.move(self.pac, pac_action)
        self.state[2,:,:][self.pac[0], self.pac[1]] = 1

        if self.gameover():
            reward = -500
            done = True
        else:
            ghosts_action = self.ghost_move()

            # win
            if np.max(self.state[1,:,:]) == 0:
                reward = 500
                done = True
            # lose
            elif self.gameover():
                reward = -500
                done = True
            else:
                # get point
                point = self.state[1,:,:][self.pac[0], self.pac[1]]
                if point >= 1:
                    self.state[1,:,:][self.pac[0], self.pac[1]] = 0
                    reward = 10 * point
                    self.my_dot += 1
                    done = False
                # nothing
                else:
                    reward = -1
                    done = False

        return self.state, reward, done, self.get_legal_action(self.pac, done)

