# import graphic
# import mygame
# pacman = mygame.Pacman(dir='smallGrid.txt')
# pacman.reset(nb_bigdot=2)
#
# g = graphic.Graphic(block_size=30)
#
# state, _ = pacman.get_state()
# g.render(state)
# print("start")
# while True:
#     g.render(state)
#     action = pacman.select_action()
#     state,reward,done,_ = pacman.make_action(action)
#     if done:
#         pacman.reset()
#         print("GAME OVER")
#     print(reward)

import numpy as np

x = np.asarray([0,2,4,6])
print (x.argmax())