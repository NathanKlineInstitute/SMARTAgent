import numpy as np
from conf import dconf
import random
from matplotlib import pyplot as plt
class simulatePong:
  def __init__ (self):
    self.court_top = 34
    self.court_bottom = 194
    self.court_redge = 159
    self.court_ledge = 0
    
    self.ball_width = 2
    self.ball_height = 4
    self.racket_width = 4
    self.racket_height = 16

    # start ball from the middle
    self.randomizeYpos = dconf['simulatedEnvParams']['random'] 
    self.ypos_ball = dconf['simulatedEnvParams']['yball']  # this corresponds to 0 index
    self.xpos_ball = 20  # this corresponds to 1 index
    self.xpos_racket = 140 # this is fixed
    self.ypos_racket = dconf['simulatedEnvParams']['yracket'] # this can change
    self.xpos_modelracket = 16 # this is fixed
    self.ypos_modelracket = 80
    self. possible_ball_ypos = [40,60,80,100,120]
    self.possible_ball_dy = [0,1,2,3]
    # create background
    self.obs = np.zeros(shape=(210,160,3))
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17
    self.leftracketx1 = self.xpos_modelracket # change mr1 to leftracket
    self.leftracketx2 = self.xpos_modelracket+self.racket_width
    self.leftrackety1 = self.court_top+self.ypos_modelracket
    self.leftrackety2 = self.court_top+self.ypos_modelracket+self.racket_height
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,0]= 213
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,1]= 130
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,2]= 74

    # create ball
    self.ballx1 = self.xpos_ball
    self.ballx2 = self.xpos_ball+self.ball_width
    self.bally1 = self.court_top+self.ypos_ball
    self.bally2 = self.court_top+self.ypos_ball+self.ball_height  
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,0]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,1]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,2]=236
    # create racket
    self.rightracketx1 = self.xpos_racket
    self.rightracketx2 = self.xpos_racket+self.racket_width
    self.rightrackety1 = self.court_top+self.ypos_racket
    self.rightrackety2 = self.court_top+self.ypos_racket+self.racket_height
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,0]= 92
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,1]= 186
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,2]= 92
    self.server = 'LRacket'
    # create ball speed or displacement
    self.ball_dx = 5  # displacement in horizontal direction
    self.ball_dy = 5  #displacement in vertical direction
    # create racket speed or displacement
    self.racket_dy = 5   # displacement of rackets. 
    # by default no reward
    self.reward =0
    self.done = 0
    # points
    self.GamePoints = []
    self.ModelPoints = []
    self.MissedTheBall = 0
    self.NewServe = 0
    self.scoreRecorded = 0
    self.fig,self.ax = plt.subplots(1,1)
    self.im = self.ax.imshow(np.zeros(shape=(210,160,3)))

  def createnewframe(self):
    self.obs = np.zeros(shape=(210,160,3))
    self.obs[self.court_top:self.court_bottom,:,0]=144
    self.obs[self.court_top:self.court_bottom,:,1]=72
    self.obs[self.court_top:self.court_bottom,:,2]=17
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,0]= 213
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,1]= 130
    #self.obs[self.mr1y:self.mr2y,self.mr1x:self.mr2x,2]= 74

  def moveball(self,xshift_ball,yshift_ball):
    self.ballx1 = self.ballx1+xshift_ball
    self.ballx2 = self.ballx2+xshift_ball
    self.bally1 = self.bally1+yshift_ball
    self.bally2 = self.bally2+yshift_ball
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,0]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,1]=236
    self.obs[self.bally1:self.bally2,self.ballx1:self.ballx2,2]=236

  def moveracket(self,yshift_racket):
    self.rightrackety1 = self.rightrackety1+yshift_racket
    self.rightrackety2 = self.rightrackety2+yshift_racket
    if self.rightrackety1>self.court_bottom-8:
      self.rightrackety1 = self.rightrackety1-yshift_racket
      self.rightrackety2 = self.rightrackety2-yshift_racket
    if self.rightrackety2<self.court_top+8:
      self.rightrackety1 = self.rightrackety1-yshift_racket
      self.rightrackety2 = self.rightrackety2-yshift_racket
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,0]= 92
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,1]= 186
    self.obs[self.rightrackety1:self.rightrackety2,self.rightracketx1:self.rightracketx2,2]= 92

  def movemodelracket(self,yshift_racket2):
    self.leftrackety1 = self.leftrackety1+yshift_racket2
    self.leftrackety2 = self.leftrackety2+yshift_racket2
    if self.leftrackety1>self.court_bottom-8:
      self.leftrackety1 = self.leftrackety1-yshift_racket2
      self.leftrackety2 = self.leftrackety2-yshift_racket2
    if self.leftrackety2<self.court_top+8:
      self.leftrackety1 = self.leftrackety1-yshift_racket2
      self.leftrackety2 = self.leftrackety2-yshift_racket2
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,0]= 213
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,1]= 130
    self.obs[self.leftrackety1:self.leftrackety2,self.leftracketx1:self.leftracketx2,2]= 74

  # xshift_ball, yshift_ball = getNextBallShift()
  def getNextBallShift(self):
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
      #print(tmp_b1y,self.court_top)
      if tmp_bally1<self.court_top: #if hit the top of the court, bounces back
        yshift_ball = self.ball_dy - tmp_bally1 + self.court_top
        tmp_bally1 = self.bally1 + yshift_ball
        tmp_bally2 = self.bally2 + yshift_ball
        self.ball_dy = -1*self.ball_dy
    else:
      yshift_ball = self.ball_dy
    # 4. check if the ball hits the racket
    if self.ball_dx>0 and tmp_ballx2>=self.rightracketx1 and tmp_ballx2<=self.court_redge: # when ball moving towards the racket controlled by the model
      if ((tmp_bally1>self.rightrackety1) and (tmp_bally1<self.rightrackety2)) or ((tmp_bally2>self.rightrackety1) and (tmp_bally2<self.rightrackety2)): 
        # if upper or lower edge of the ball is within the range of the racket
        xshift_ball = self.ball_dx + self.rightracketx1-tmp_ballx2
        if tmp_bally2>(self.rightrackety1 + 0.5*self.racket_height): # if the ball hits the lower half of the racket
          yshift_ball = 1 + self.ball_dy
        elif tmp_bally2<(self.rightrackety1 + 0.5*self.racket_height):
          yshift_ball = -1 + self.ball_dy
        self.ball_dx = -1*self.ball_dx
      else:
        if self.scoreRecorded==0:
          self.GamePoints.append(1)
          self.MissedTheBall = 1
          print('Right player missed the ball')
          print('Scores: ', len(self.GamePoints),len(self.ModelPoints))
          print(self.ballx1,self.ballx2,self.bally1,self.bally2,self.rightracketx1,self.rightracketx2,self.rightrackety1,self.rightrackety2,xshift_ball,yshift_ball)
          self.reward = -1
          self.scoreRecorded = 1 
    elif self.ball_dx<0 and tmp_ballx1<=self.leftracketx2 and tmp_ballx1>=self.court_ledge: # when ball moving towards the racket controlled internally.
      if ((tmp_bally1>self.leftrackety1) and (tmp_bally1<self.leftrackety2)) or ((tmp_bally2>self.leftrackety1) and (tmp_bally2<self.leftrackety2)):
        # if upper or lower edge of the ball is within the range of the racket
        xshift_ball = self.ball_dx + self.leftracketx2-tmp_ballx1
        if tmp_bally2>(self.leftrackety1 + 0.5*self.racket_height): # if the ball hits the lower half of the racket
          yshift_ball = 1 + self.ball_dy
        elif tmp_bally2<(self.leftrackety1 + 0.5*self.racket_height):
          yshift_ball = -1 + self.ball_dy
        self.ball_dx = -1*self.ball_dx
      else:
        if self.scoreRecorded==0:
          self.ModelPoints.append(1)
          self.MissedTheBall = 1
          print('Left player missed the ball')
          print('Scores: ', len(self.GamePoints),len(self.ModelPoints))
          print(self.ballx1,self.ballx2,self.bally1,self.bally2,self.leftracketx1,self.leftracketx2,self.leftrackety1,self.leftrackety2,xshift_ball,yshift_ball)
          self.reward = 1
          self.scoreRecorded = 1
    else:
      xshift_ball = self.ball_dx
    TotalPoints = len(self.ModelPoints) + len(self.GamePoints)
    if self.MissedTheBall:     #reset the location of the ball as well as self.ball_dx and self.ball_dy
      print(tmp_ballx1,tmp_ballx2)
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
          self.ball_dx = 5
        elif self.server=='RRacket':
          self.ball_dx = -5
    return xshift_ball, yshift_ball


  def step(self,action):
    stepsize = self.racket_dy
    if action==3:
      yshift_racket=stepsize
    elif action==4:
      yshift_racket=-1*stepsize
    else:
      yshift_racket=0
    self.createnewframe()
    #randaction = random.randint(3,4)
    #if randaction==3: rand_yshift = stepsize
    #else: rand_yshift = -stepsize
    if (self.leftrackety1+0.5*self.racket_height)>(self.bally1+0.5*self.ball_height):
      rand_yshift = -stepsize
    elif (self.leftrackety1+0.5*self.racket_height)<(self.bally1+0.5*self.ball_height):
      rand_yshift = stepsize
    else:
      rand_yshift = 0
    self.movemodelracket(rand_yshift) # intead of random shift, yshift should be based on projection
    self.moveracket(yshift_racket) # this should be always based on Model/User

    xshift_ball, yshift_ball = self.getNextBallShift() # needs ball coords, both rackets' coordinates as well as boundaries.
    print(self.MissedTheBall,self.NewServe,self.scoreRecorded, xshift_ball, yshift_ball)
    #print(self.b1x,self.b2x,self.b1y,self.b2y)
    if self.NewServe==1:
      self.ballx1 = self.xpos_ball
      self.ballx2 = self.xpos_ball+self.ball_width
      self.bally1 = self.court_top+self.ypos_ball
      self.bally2 = self.court_top+self.ypos_ball+self.ball_height
      if (len(self.GamePoints)>1 and len(self.GamePoints)%20==0) or (len(self.ModelPoints) and len(self.ModelPoints)%20==0):
        self.done = 1
      else:
        self.done = 0
    self.moveball(xshift_ball=xshift_ball, yshift_ball=yshift_ball) # this should be computed internally  
    self.obs = self.obs.astype(np.uint8)
    self.im.set_data(self.obs.astype(np.uint8))
    self.fig.canvas.draw_idle()
    plt.pause(0.1)
    return self.obs, self.reward, self.done




#when the ball is moving in positive X dir then should be checked for hitting the Right racket.
#If the ball hits the right racket: look at the angle and flip the angle.
#else if the ball reaches the right edge, reset the ball.
#else if the ball hits the upper or lower edge, look at the angle and flip the angle.

#when the ball is moving in negative X dir then should be checked for hitting the left racket.
#If the ball hits the left racket: look at the angle and flip the angle.
#else if the ball hits the left edge, reset the ball.
#else if the ball hits the upper of lower edge, look at the angle and flip the angle.
