import numpy as np
from conf import dconf
import random
from matplotlib import pyplot as plt
    
class simplePong:
  def __init__  (self, seed=1234):
    print('spp A')
    random.seed(seed)
    print('spp B')    
    self.createcourt()
    print('spp C')    
    self.obs = np.zeros(shape=(210,160,3)) # this is the image (observation) frame
    print('spp D')    
    self.createnewframe()
    print('spp D')
    self.createball() # create ball
    print('spp E')    
    self.createrackets() # create rackets
    print('spp F')    
    # by default no reward
    self.reward =0
    # points
    self.TotalHits = 0
    self.TotalMissed = 0
    self.MissedTheBall = 0
    self.NewServe = 0
    self.scoreRecorded = 0
    print('spp G')    
    self.createFigure()
    print('spp H')    

  def createFigure (self):
    self.fig,self.ax = plt.subplots(1,1)
    self.im = self.ax.imshow(np.zeros(shape=(210,160,3)))
    self.scorestr = self.ax.text(1, 20, 'M,H:0,0', style='normal', color='lightgreen', size=28)

  def createcourt (self):
    self.court_top = 34
    self.court_bottom = 194
    self.court_redge = 159
    self.court_ledge = 0    

  def createball (self):
    # ball position
    self.ypos_ball = dconf['simulatedEnvParams']['yball']  # this corresponds to 0 index
    self.xpos_ball = 0  # this corresponds to 1 index
    self.possible_ball_ypos = dconf['simulatedEnvParams']['possible_ball_ypos']
    self.possible_ball_dy = dconf['simulatedEnvParams']['possible_ball_dy']
    self.possible_ball_dx = dconf['simulatedEnvParams']['possible_ball_dx']    
    # start ball from the middle
    self.randomizeYpos = dconf['simulatedEnvParams']['random'] 
    if self.randomizeYpos:
      self.ypos_ball = random.choice(self.possible_ball_ypos)
      print('randomize y pos, start ball at y = ', self.ypos_ball)
    self.wiggle = dconf['wiggle']
    self.ball_width = 4
    self.ball_height = 4    
    self.ballx1 = self.xpos_ball
    self.ballx2 = self.xpos_ball+self.ball_width
    self.bally1 = self.court_top+self.ypos_ball
    self.bally2 = self.court_top+self.ypos_ball+self.ball_height  
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,0]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,1]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,2]=236
    # create ball speed or displacement
    self.ball_dx = 1  # displacement in horizontal direction
    self.ball_dy = 1  #displacement in vertical direction

  def createrackets (self):
    self.racket_width = 4
    self.racket_height = 16
    # racket positions
    self.xpos_racket = 140 # this is fixed
    self.ypos_racket = dconf['simulatedEnvParams']['yracket'] # this can change
    # right racket
    self.rightracketx1 = self.xpos_racket
    self.rightracketx2 = self.xpos_racket+self.racket_width
    self.rightrackety1 = self.court_top+self.ypos_racket
    self.rightrackety2 = self.court_top+self.ypos_racket+self.racket_height
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,0]= 92
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,1]= 186
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,2]= 92    
    # create racket speed or displacement
    self.racket_dy = dconf['simulatedEnvParams']['racket_dy'] # displacement of rackets.
    
  def createnewframe (self):
    self.obs.fill(0)
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17

  def moveball (self,xshift_ball,yshift_ball):
    self.ballx1 += xshift_ball
    self.ballx2 += xshift_ball
    self.bally1 += yshift_ball
    self.bally2 += yshift_ball
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,0]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,1]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,2]=236

  def moveracket (self,yshift_racket):
    self.rightrackety1 += yshift_racket
    self.rightrackety2 += yshift_racket
    if self.rightrackety1 > self.court_bottom - self.racket_height:
      self.rightrackety1 -= yshift_racket
      self.rightrackety2 -= yshift_racket
    if self.rightrackety2 < self.court_top + self.racket_height:
      self.rightrackety1 -= yshift_racket
      self.rightrackety2 -= yshift_racket
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,0]= 92
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,1]= 186
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,2]= 92

  # xshift_ball, yshift_ball = getNextBallShift()
  def getNextBallShift (self, right_racket_yshift):
    # ball position is defined by self.b1x, self.b2x, self.b1y and self.b2y
    # right racket position is defined by self.r1y, self.r2y, self.r1x and self.r2x. Both self.r1x and self.r2x are fixed.
    # court coordinates are self.court_top = 34, self.court_bottom = 194, self.court_ledge = 0, self.court_redge = 159
    # direction can be checked by looking at the sign of self.ball_dx and self.ball_dy
    # 1. make a temp move
    self.reward = 0
    if self.ballx2<self.rightracketx1 and self.NewServe:
      # why need to set missedball==0 here? can't you set it after it's set to 1 at end of this function?
      # and why does missedtheball need to persist outside of this function - it's only used within this function
      # answer: probably because ball can keep moving beyond rackets and do not want to count score twice ... 
      self.MissedTheBall = 0
      self.NewServe = 0
      self.scoreRecorded = 0
    tmp_ballx1 = self.ballx1 + self.ball_dx
    tmp_ballx2 = self.ballx2 + self.ball_dx
    tmp_bally1 = self.bally1 + self.ball_dy
    tmp_bally2 = self.bally2 + self.ball_dy
    xshift_ball = self.ball_dx
    yshift_ball = self.ball_dy
    # check if the ball hits the left edge
    if self.ball_dx<0: # moving leftwards
      if tmp_ballx1<0: # if hit the bottom of the court, bounces back
        xshift_ball = self.ball_dx - tmp_ballx1
        self.ball_dy = np.sign(self.ball_dy) * abs(random.choice(self.possible_ball_dy))
        y_shift_ball = self.ball_dy
        self.ball_dx *= -1        
    # 2. check if the ball hits upper edge or lower edge
    if self.ball_dy>0: # moving downwards
      if tmp_bally2>=self.court_bottom: # if hit the bottom of the court, bounces back
        yshift_ball = self.ball_dy + self.court_bottom - tmp_bally2
        tmp_bally1 = self.bally1 + yshift_ball
        tmp_bally2 = self.bally2 + yshift_ball
        self.ball_dy = -1*self.ball_dy
    elif self.ball_dy<0: # moving upwards
      if tmp_bally1<=self.court_top: #if hit the top of the court, bounces back
        yshift_ball = self.ball_dy - tmp_bally1 + self.court_top
        tmp_bally1 = self.bally1 + yshift_ball
        tmp_bally2 = self.bally2 + yshift_ball
        self.ball_dy = -1*self.ball_dy
    else:
      yshift_ball = self.ball_dy
    # 4. check if the ball hits the racket
    # when ball moving towards the (right) racket controlled by the model
    if self.ball_dx>0 and tmp_ballx2>=self.rightracketx1 and tmp_ballx2<=self.court_redge and self.MissedTheBall==0: 
      if (tmp_bally1>=self.rightrackety1 and tmp_bally1<=self.rightrackety2) or (tmp_bally2>=self.rightrackety1 and tmp_bally2<=self.rightrackety2): 
        # if upper or lower edge of the ball is within the range of the racket
        xshift_ball = self.ball_dx + self.rightracketx1-tmp_ballx2
        self.ball_dy = np.sign(self.ball_dy) * abs(random.choice(self.possible_ball_dy))
        y_shift_ball = self.ball_dy
        self.ball_dx *= -1
        self.TotalHits += 1
        self.reward = 1
      elif dconf['simulatedEnvParams']['top_bottom_rule'] and right_racket_yshift < 0 and abs(tmp_bally2 - self.rightrackety1) <= 2:
        print('hit top R')
        xshift_ball = self.ball_dx + self.rightracketx1 - tmp_ballx2
        self.ball_dy = np.sign(self.ball_dy) * abs(random.choice(self.possible_ball_dy)) * 2        
        y_shift_ball = self.ball_dy
        self.ball_dx *= -1
        self.TotalHits += 1
        self.reward = 1        
      elif dconf['simulatedEnvParams']['top_bottom_rule'] and right_racket_yshift > 0 and abs(tmp_bally1 - self.rightrackety2) <= 2:
        print('hit bottom R')
        xshift_ball = self.ball_dx + self.rightracketx1 - tmp_ballx2
        self.ball_dy = np.sign(self.ball_dy) * abs(random.choice(self.possible_ball_dy)) * 2
        y_shift_ball = self.ball_dy
        self.ball_dx *= -1
        self.TotalHits += 1
        self.reward = 1        
      else:
        if self.scoreRecorded==0 and tmp_ballx1>self.rightracketx2:
          self.TotalMissed += 1
          self.MissedTheBall = 1
          if not dconf['simulatedEnvParams']['dodraw']:
            print('Player missed the ball')
            print('Hits: ', self.TotalHits, 'Missed: ',self.TotalMissed)            
            print('Ball (projected):',tmp_ballx1,tmp_ballx2,tmp_bally1,tmp_bally2)
            print('Racket:',self.rightracketx1,self.rightracketx2,self.rightrackety1,self.rightrackety2)
          self.reward = -1
          self.scoreRecorded = 1 
    else:
      xshift_ball = self.ball_dx
    if self.MissedTheBall:     #reset the location of the ball as well as self.ball_dx and self.ball_dy
      if tmp_ballx1<self.court_ledge or tmp_ballx2>self.court_redge:
        self.NewServe = 1
        xshift_ball = 0
        yshift_ball = 0
        self.ball_dy = random.choice(self.possible_ball_dy)
        self.xpos_ball = 0
        self.ypos_ball = random.choice(self.possible_ball_ypos)
        self.ball_dx = random.choice(self.possible_ball_dx)
    return xshift_ball, yshift_ball

  def step (self,action):
    # one step of game activity
    stepsize = self.racket_dy

    if action==3:
      right_racket_yshift = stepsize
    elif action==4:
      right_racket_yshift = -stepsize
    elif action==1:
      right_racket_yshift=0
    else: # invalid action means right paddle follows the ball (not done using learning neuronal network model)
     ballmidY = self.bally1 + 0.5 * self.ball_height
     if ballmidY > self.rightrackety2 - self.wiggle: # if ball is below bottom of racket
       right_racket_yshift = stepsize # down
     elif self.bally2 < self.rightrackety1 + self.wiggle: # if ball is above top of racket
       right_racket_yshift = -stepsize # up
     else:
       right_racket_yshift = 0
      
    self.createnewframe()

    # this rule moves paddle only when it does not overlap with ball along vertical axis +/- self.wiggle
    # when using self.wiggle of ~1/2 paddle height, it introduces oscillations in paddle as it tracks the ball
    self.moveracket(right_racket_yshift) # this should be always based on Model/User
    # needs ball coords, both rackets' coordinates as well as boundaries.
    xshift_ball, yshift_ball = self.getNextBallShift(right_racket_yshift) 
    if self.NewServe==1:
      self.ballx1 = self.xpos_ball
      self.ballx2 = self.xpos_ball+self.ball_width
      self.bally1 = self.court_top+self.ypos_ball
      self.bally2 = self.court_top+self.ypos_ball+self.ball_height
    self.moveball(xshift_ball, yshift_ball) # this should be computed internally  
    self.obs = self.obs.astype(np.uint8)
    if dconf['simulatedEnvParams']['dodraw']:
      self.im.set_data(self.obs)#.astype(np.uint8))
      self.drawscore()        
      plt.pause(0.0001)
    return self.obs, self.reward

  def drawscore (self):
    self.scorestr.set_text('M,H:'+str(self.TotalMissed)+','+str(self.TotalHits))

  def reset (self):
    print('WARNING: empty reset')

#when the ball is moving in positive X dir then should be checked for hitting the Right racket.
#If the ball hits the right racket: look at the angle and flip the angle.
#else if the ball reaches the right edge, reset the ball.
#else if the ball hits the upper or lower edge, look at the angle and flip the angle.

def testsim (nstep=10000):
  # test the simulated pong with nstep
  pong = simplePong()
  for i in range(nstep):
    #randaction = random.choice([3,4,1])
    obs, reward = pong.step(-1)#randaction)
    

if __name__ == '__main__':
  testsim()
        
