import numpy as np
from conf import dconf
import random
class simulatePong:
  def __init__ (self):
    self.court_top = 34
    self.court_bottom = 194
    self.ball_width = 2
    self.ball_height = 4
    self.racket_width = 4
    self.racket_height = 16
    # start ball from the middle
    self.ypos_ball = dconf['simulatedEnvParams']['yball']  # this corresponds to 0 index
    self.xpos_ball = 20  # this corresponds to 1 index
    self.xpos_racket = 140 # this is fixed
    self.ypos_racket = dconf['simulatedEnvParams']['yracket'] # this can change
    self.xpos_modelracket = 16 # this is fixed
    self.ypos_modelracket = 80
    # create background
    self.obs = np.zeros(shape=(210,160,3))
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17
    self.mr1x = self.xpos_modelracket
    self.mr2x = self.xpos_modelracket+self.racket_width
    self.mr1y = self.court_top+self.ypos_modelracket
    self.mr2y = self.court_top+self.ypos_modelracket+self.racket_height
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,0]= 213
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,1]= 130
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,2]= 74

    # create ball
    self.b1x = self.xpos_ball
    self.b2x = self.xpos_ball+self.ball_width
    self.b1y = self.court_top+self.ypos_ball
    self.b2y = self.court_top+self.ypos_ball+self.ball_height  
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,0]=236
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,1]=236
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,2]=236
    # create racket
    self.r1x = self.xpos_racket
    self.r2x = self.xpos_racket+self.racket_width
    self.r1y = self.court_top+self.ypos_racket
    self.r2y = self.court_top+self.ypos_racket+self.racket_height
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,0]= 92
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,1]= 186
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,2]= 92
    # by default no reward
    self.reward =0
    self.done = 0

  def createnewframe(self):
    self.obs = np.zeros(shape=(210,160,3))
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,0]= 213
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,1]= 130
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,2]= 74

  def moveball(self,xshift_ball,yshift_ball):
    self.b1x = self.b1x+xshift_ball
    self.b2x = self.b2x+xshift_ball
    self.b1y = self.b1y+yshift_ball
    self.b2y = self.b2y+yshift_ball
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,0]=236
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,1]=236
    self.obs[self.b1y:self.b2y,self.b1x:self.b2x,2]=236

  def moveracket(self,yshift_racket):
    self.r1y = self.r1y+yshift_racket
    self.r2y = self.r2y+yshift_racket
    if self.r1y>self.court_bottom-8:
      self.r1y = self.r1y-yshift_racket
      self.r2y = self.r2y-yshift_racket
    if self.r2y<self.court_top+8:
      self.r1y = self.r1y-yshift_racket
      self.r2y = self.r2y-yshift_racket
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,0]= 92
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,1]= 186
    self.obs[self.r1y:self.r2y,self.r1x:self.r2x,2]= 92

  def movemodelracket(self,yshift_racket2):
    self.mr1y = self.mr1y+yshift_racket2
    self.mr2y = self.mr2y+yshift_racket2
    if self.mr1y>self.court_bottom-8:
      self.mr1y = self.mr1y-yshift_racket2
      self.mr2y = self.mr2y-yshift_racket2
    if self.mr2y<self.court_top+8:
      self.mr1y = self.mr1y-yshift_racket2
      self.mr2y = self.mr2y-yshift_racket2
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,0]= 213
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,1]= 130
    self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,2]= 74

  def step(self,action):
    if action==3:
      yshift_racket=10
    elif action==4:
      yshift_racket=-10
    else:
      yshift_racket=0
    self.createnewframe()
    randaction = random.randint(3,4)
    if randaction==3: rand_yshift = 10
    else: rand_yshift = -10 
    self.movemodelracket(rand_yshift)
    self.moveracket(yshift_racket)
    self.moveball(xshift_ball=3, yshift_ball=0)
    if self.b2x>=self.r1x:
      if ((self.b1y>self.r1y) and (self.b1y<self.r2y)) or ((self.b2y>self.r1y) and (self.b2y<self.r2y)): # if upper or lower edge of the ball is within the range of the racket 
        if self.done==0:
          self.reward = 1
          self.b1x = self.xpos_ball
          self.b2x = self.xpos_ball+self.ball_width
          self.b1y = self.court_top+self.ypos_ball
          self.b2y = self.court_top+self.ypos_ball+self.ball_height
          self.done = 1
      else:
        if self.done==0:
          self.reward = -1
          self.done = 1
        else:
          self.reward = 0
    else:
      self.reward = 0
    if self.b2x>self.r2x+4:
      self.b1x = self.xpos_ball
      self.b2x = self.xpos_ball+self.ball_width
      self.b1y = self.court_top+self.ypos_ball
      self.b2y = self.court_top+self.ypos_ball+self.ball_height
      self.done = 0
    self.obs = self.obs.astype(np.uint8)
    return self.obs, self.reward, self.done
