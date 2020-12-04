import numpy as np
from conf import dconf
import random
from matplotlib import pyplot as plt

"""
class ball:
  def __init__ (self,x, y, width, height):
    self.x = x
    self.y = y
    self.width = width
    self.height = height
    # these variables are for keeping track of our velocity on both the
    # X and Y axis, since the ball can move in two dimensions
    self.dy = 0
    self.dx = 0
  def collides (self,paddle):
    #Expects a paddle as an argument and returns true or false, depending
    #on whether their rectangles overlap.
    # first, check to see if the left edge of either is farther to the right
    # than the right edge of the other
    if self.x > paddle.x + paddle.width or paddle.x > self.x + self.width:
      return False
    # then check to see if the bottom edge of either is higher than the top
    # edge of the other
    if self.y > paddle.y + paddle.height or paddle.y > self.y + self.height:
      return False
    # if the above aren't true, they're overlapping
    return True
  def reset(self):
    # Places the ball in the middle of the screen, with no movement.
    self.x = VIRTUAL_WIDTH / 2 - 2
    self.y = VIRTUAL_HEIGHT / 2 - 2
    self.dx = 0
    self.dy = 0
  def update(self, dt)
    self.x = self.x + self.dx * dt
    self.y = self.y + self.dy * dt
  def render(self):
    pass
    #love.graphics.rectangle('fill', self.x, self.y, self.width, self.height)
"""
    
class simulatePong:
  def __init__  (self, seed=1234):
    random.seed(seed)
    self.createcourt()
    self.obs = np.zeros(shape=(210,160,3)) # this is the image (observation) frame
    self.createnewframe()
    self.createball() # create ball
    self.createrackets() # create rackets
    # by default no reward
    self.reward =0
    self.done = 0
    # points
    self.GamePoints = 0
    self.ModelPoints = 0
    self.MissedTheBall = 0
    self.NewServe = 0
    self.scoreRecorded = 0
    self.createFigure()

  def createFigure (self):
    self.fig,self.ax = plt.subplots(1,1)
    self.im = self.ax.imshow(np.zeros(shape=(210,160,3)))
    self.scoreleft = self.ax.text(30, 20, str(self.GamePoints), style='normal',color='orange',size=32)
    self.scoreright = self.ax.text(130, 20, str(self.ModelPoints), style='normal',color='lightgreen',size=32)

  def createcourt (self):
    self.court_top = 34
    self.court_bottom = 194
    self.court_redge = 159
    self.court_ledge = 0    

  def createball (self):
    # ball position
    self.ypos_ball = dconf['simulatedEnvParams']['yball']  # this corresponds to 0 index
    self.xpos_ball = 20  # this corresponds to 1 index        
    # start ball from the middle
    self.randomizeYpos = dconf['simulatedEnvParams']['random']     
    self.ball_width = 2
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
    self. possible_ball_ypos = [40,60,80,100,120]
    self.possible_ball_dy = [1,1,1,1,1,1,2,2,2,2,3,3,3]
    self.possible_ball_dx = [1,1,1,1,1,1,2,2,2,2,3,3,3]

  def createrackets (self):
    self.racket_width = 4
    self.racket_height = 16
    # racket positions
    self.xpos_racket = 140 # this is fixed
    self.ypos_racket = dconf['simulatedEnvParams']['yracket'] # this can change
    self.xpos_modelracket = 16 # this is fixed
    self.ypos_modelracket = 80    
    # left racket
    self.leftracketx1 = self.xpos_modelracket # change mr1 to leftracket
    self.leftracketx2 = self.xpos_modelracket+self.racket_width
    self.leftrackety1 = self.court_top+self.ypos_modelracket
    self.leftrackety2 = self.court_top+self.ypos_modelracket+self.racket_height
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,0]= 213
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,1]= 130
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,2]= 74
    # right racket
    self.rightracketx1 = self.xpos_racket
    self.rightracketx2 = self.xpos_racket+self.racket_width
    self.rightrackety1 = self.court_top+self.ypos_racket
    self.rightrackety2 = self.court_top+self.ypos_racket+self.racket_height
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,0]= 92
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,1]= 186
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,2]= 92    
    # create racket speed or displacement
    self.racket_dy = 5   # displacement of rackets.
    self.server = 'LRacket' # which racket serving    
    
  def createnewframe (self):
    self.obs.fill(0)
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,0]= 213
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,1]= 130
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,2]= 74

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

  def movemodelracket (self,yshift_racket2):
    self.leftrackety1 += yshift_racket2
    self.leftrackety2 += yshift_racket2
    if self.leftrackety1 > self.court_bottom - self.racket_height:
      self.leftrackety1 -= yshift_racket2
      self.leftrackety2 -= yshift_racket2
    if self.leftrackety2 < self.court_top + self.racket_height:
      self.leftrackety1 -= yshift_racket2
      self.leftrackety2 -= yshift_racket2
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,0]= 213
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,1]= 130
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,2]= 74

  # xshift_ball, yshift_ball = getNextBallShift()
  def getNextBallShift (self):
    # ball position is defined by self.b1x, self.b2x, self.b1y and self.b2y
    # right racket position is defined by self.r1y, self.r2y, self.r1x and self.r2x. Both self.r1x and self.r2x are fixed.
    # left racket position is defined by self.mr1y, self.mr2y, self.mr1x and self.mr2x. Both self.mr1x and self.mr2x are fixed.
    # court coordinates are self.court_top = 34, self.court_bottom = 194, self.court_ledge = 0, self.court_redge = 159
    # self.ball_dir = (1,1)
    # direction can be checked by looking at the sign of self.ball_dx and self.ball_dy
    # 1. make a temp move
    self.reward = 0
    if self.ballx1>self.leftracketx2 and self.ballx2<self.rightracketx1 and self.NewServe:
      self.MissedTheBall = 0
      self.NewServe = 0
      self.scoreRecorded = 0
    tmp_ballx1 = self.ballx1 + self.ball_dx
    tmp_ballx2 = self.ballx2 + self.ball_dx
    tmp_bally1 = self.bally1 + self.ball_dy
    tmp_bally2 = self.bally2 + self.ball_dy
    xshift_ball = self.ball_dx
    yshift_ball = self.ball_dy
    # 2. check if the ball hits upper edge or lower edge
    if self.ball_dy>0: # moving downwards
      if tmp_bally2>self.court_bottom: # if hit the bottom of the court, bounces back
        yshift_ball = self.ball_dy + self.court_bottom - tmp_bally2
        tmp_bally1 = self.bally1 + yshift_ball
        tmp_bally2 = self.bally2 + yshift_ball
        self.ball_dy = -1*self.ball_dy
    elif self.ball_dy<0: # moving upwards
      if tmp_bally1<self.court_top: #if hit the top of the court, bounces back
        yshift_ball = self.ball_dy - tmp_bally1 + self.court_top
        tmp_bally1 = self.bally1 + yshift_ball
        tmp_bally2 = self.bally2 + yshift_ball
        self.ball_dy = -1*self.ball_dy
    else:
      yshift_ball = self.ball_dy
    # 4. check if the ball hits the racket
    # when ball moving towards the racket controlled by the model
    if self.ball_dx>0 and tmp_ballx2>=self.rightracketx1 and tmp_ballx2<=self.court_redge and self.MissedTheBall==0: 
      if ((tmp_bally1>self.rightrackety1) and (tmp_bally1<self.rightrackety2)) or ((tmp_bally2>self.rightrackety1) and (tmp_bally2<self.rightrackety2)): 
        # if upper or lower edge of the ball is within the range of the racket
        xshift_ball = self.ball_dx + self.rightracketx1-tmp_ballx2
        """
        if tmp_bally2>(self.rightrackety1 + 0.5*self.racket_height): # if the ball hits the lower half of the racket
          yshift_ball = 1 + self.ball_dy
        elif tmp_bally2<(self.rightrackety1 + 0.5*self.racket_height):
          yshift_ball = -1 + self.ball_dy
        """
        if self.ball_dy < 0:
          y_shift_ball = -random.choice(self.possible_ball_dy)
        else:
          y_shift_ball = random.choice(self.possible_ball_dy) 
        self.ball_dx *= -1
      else:
        if self.scoreRecorded==0:
          self.GamePoints += 1
          self.MissedTheBall = 1
          #print('Right player missed the ball')
          #print('Scores: ', self.GamePoints,self.ModelPoints)
          self.reward = -1
          self.scoreRecorded = 1 
    elif self.ball_dx<0 and tmp_ballx1<=self.leftracketx2 and tmp_ballx1>=self.court_ledge and self.MissedTheBall==0:
      # when ball moving towards the racket controlled internally.
      if ((tmp_bally1>self.leftrackety1) and (tmp_bally1<self.leftrackety2)) or ((tmp_bally2>self.leftrackety1) and (tmp_bally2<self.leftrackety2)):
        # if upper or lower edge of the ball is within the range of the racket
        xshift_ball = self.ball_dx + self.leftracketx2-tmp_ballx1
        """
        if tmp_bally2>(self.leftrackety1 + 0.5*self.racket_height): # if the ball hits the lower half of the racket
          yshift_ball = 1 + self.ball_dy
        elif tmp_bally2<(self.leftrackety1 + 0.5*self.racket_height):
          yshift_ball = -1 + self.ball_dy
        """
        if self.ball_dy < 0:
          y_shift_ball = -random.choice(self.possible_ball_dy)
        else:
          y_shift_ball = random.choice(self.possible_ball_dy)           
        self.ball_dx *= -1
      else:
        if self.scoreRecorded==0:
          self.ModelPoints += 1
          self.MissedTheBall = 1
          #print('Left player missed the ball')
          #print('Scores: ', self.GamePoints,self.ModelPoints)
          self.reward = 1
          self.scoreRecorded = 1
    else:
      xshift_ball = self.ball_dx
    TotalPoints = self.ModelPoints + self.GamePoints
    if self.MissedTheBall:     #reset the location of the ball as well as self.ball_dx and self.ball_dy
      if tmp_ballx1<self.court_ledge or tmp_ballx2>self.court_redge:
        self.NewServe = 1
        xshift_ball = 0
        yshift_ball = 0
        self.ball_dy = random.choice(self.possible_ball_dy)
        self.xpos_ball = 80
        self.ypos_ball = random.choice(self.possible_ball_ypos)
        if TotalPoints%5==0: #after every 5 points,change the server
          if self.server == 'LRacket':
            self.server = 'RRacket'
          else:
            self.server = 'LRacket'
        if self.server=='LRacket':
          self.ball_dx = random.choice(self.possible_ball_dx)
        elif self.server=='RRacket':
          self.ball_dx = -random.choice(self.possible_ball_dx)
    return xshift_ball, yshift_ball

  """
  def predictBallRacketYIntercept(self, xpos1, ypos1, xpos2, ypos2):
    if ((xpos1==-1) or (xpos2==-1)):
      predY = -1
    else:
      deltax = xpos2-xpos1
      if deltax<=0:
        predY = -1
      else:
        if ypos1<0:
          predY = -1
        else:
          NB_intercept_steps = np.ceil((120.0 - xpos2)/deltax)
          deltay = ypos2-ypos1
          predY_nodeflection = ypos2 + (NB_intercept_steps*deltay)
          if predY_nodeflection<0:
            predY = -1*predY_nodeflection
          elif predY_nodeflection>160:
            predY = predY_nodeflection-160
          else:
            predY = predY_nodeflection
    return predY
  """
  
  def step (self,action):
    # one step of game activity
    stepsize = self.racket_dy

    if action==3:
      yshift_racket = stepsize
    elif action==4:
      yshift_racket = -stepsize
    elif action==1:
      yshift_racket=0
    else: # invalid action means right paddle follows the ball (not done using learning neuronal network model)
     ballmidY = self.bally1 + 0.5 * self.ball_height
     wiggle = 6
     if ballmidY > self.rightrackety2 - wiggle:
       yshift_racket = stepsize
     elif ballmidY < self.rightrackety1 + wiggle:
       yshift_racket = -stepsize
     else:
       yshift_racket = 0
      
    self.createnewframe()

    #predY = self.predictBallRacketYIntercept() #(self.ballx1+self.ballx2)/2., (self.bally1+self.bally2)/2., \
    #self.ballx2, self.bally2)

    # this rule moves paddle only when it does not overlap with ball along vertical axis +/- wiggle
    # when using wiggle of ~1/2 paddle height, it introduces oscillations in paddle as it tracks the ball
    ballmidY = self.bally1 + 0.5 * self.ball_height
    wiggle = 6
    if ballmidY > self.leftrackety2 - wiggle:
      left_yshift = stepsize
    elif ballmidY < self.leftrackety1 + wiggle:
      left_yshift = -stepsize
    else:
      left_yshift = 0

    """
    if (self.leftrackety1+0.5*self.racket_height)>(self.bally1+0.5*self.ball_height):
      left_yshift = -stepsize
    elif (self.leftrackety1+0.5*self.racket_height)<(self.bally1+0.5*self.ball_height):
      left_yshift = stepsize
    else:
      left_yshift = 0
    """
      
    self.movemodelracket(left_yshift) # intead of random shift, yshift should be based on projection
    self.moveracket(yshift_racket) # this should be always based on Model/User
    xshift_ball, yshift_ball = self.getNextBallShift() # needs ball coords, both rackets' coordinates as well as boundaries.
    if self.NewServe==1:
      self.ballx1 = self.xpos_ball
      self.ballx2 = self.xpos_ball+self.ball_width
      self.bally1 = self.court_top+self.ypos_ball
      self.bally2 = self.court_top+self.ypos_ball+self.ball_height
      if (self.GamePoints>1 and self.GamePoints%20==0) or (self.ModelPoints and self.ModelPoints%20==0):
        self.done = 1
      else:
        self.done = 0
    self.moveball(xshift_ball, yshift_ball) # this should be computed internally  
    self.obs = self.obs.astype(np.uint8)
    self.im.set_data(self.obs.astype(np.uint8))
    self.drawscore()        
    self.fig.canvas.draw_idle()
    plt.pause(0.0001)
    #plt.ion()
    return self.obs, self.reward, self.done

  def drawscore (self):
    self.scoreleft.set_text(str(self.GamePoints))
    self.scoreright.set_text(str(self.ModelPoints))

#when the ball is moving in positive X dir then should be checked for hitting the Right racket.
#If the ball hits the right racket: look at the angle and flip the angle.
#else if the ball reaches the right edge, reset the ball.
#else if the ball hits the upper or lower edge, look at the angle and flip the angle.

#when the ball is moving in negative X dir then should be checked for hitting the left racket.
#If the ball hits the left racket: look at the angle and flip the angle.
#else if the ball hits the left edge, reset the ball.
#else if the ball hits the upper of lower edge, look at the angle and flip the angle.

def testsim (nstep=10000):
  # test the simulated pong with nstep
  pong = simulatePong()
  for i in range(nstep):
    #randaction = random.choice([3,4,1])
    obs, reward, done = pong.step(-1)#randaction)
    

if __name__ == '__main__':
  testsim()
        
