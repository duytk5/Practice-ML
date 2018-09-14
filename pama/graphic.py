import pygame
import sys
import numpy as np

class Graphic():
    def __init__(self, block_size=15):
        SCREEN_SIZE = WIDTH, HEIGHT = (700, 500)
        self.bg = (0, 0, 0)
        self.wall = (7, 43, 226)
        self.pac = (247, 160, 9)
        self.ghost = (250, 20, 6)
        self.dot = (155, 229, 8)
        self.block_size = block_size
        
        pygame.init()
        self.screen = pygame.display.set_mode(SCREEN_SIZE)
        self.clock = pygame.time.Clock()
        
    def draw(self, color, mat, rect=True):
        h, w = mat.shape
        for x in range(h):
            for y in range(w):
                if mat[x, y] >= 1:
                    if rect:
                        pygame.draw.rect(self.screen, color, (y*self.block_size, x*self.block_size, self.block_size, self.block_size), 0)
                    else:
                        if mat[x, y] > 1:
                            pygame.draw.circle(self.screen, self.ghost, (int((y+0.5)*self.block_size), int((x+0.5)*self.block_size)), int(self.block_size/2.7), 0)
                        else:
                            pygame.draw.circle(self.screen, color, (int((y+0.5)*self.block_size), int((x+0.5)*self.block_size)), int(self.block_size/4.0), 0)
                    

    def render(self, state):
      
        self.screen.fill(self.bg)
        self.draw(self.wall, state[0,:,:])
        self.draw(self.pac, state[2,:,:])
        self.draw(self.ghost, state[3,:,:])
        self.draw(self.dot, state[1,:,:], False)
        
        pygame.display.update()
        self.clock.tick(50)
