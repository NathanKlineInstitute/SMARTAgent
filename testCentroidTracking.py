from centroidtracker import CentroidTracker
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from scipy import ndimage
from aigame import AIGame
import time
import matplotlib.gridspec as gridspec
from matplotlib import animation
import anim
from collections import OrderedDict
import copy
from skimage.transform import downscale_local_mean, rescale, resize

AIGame = AIGame()
#for _ in range(20):
rewards, epCount, proposed_actions, total_hits, Racket_pos, Ball_pos = AIGame.playGame(actions=[3], epCount = 0)

def getObjectsBoundingBoxes(frame):
  mask = frame > np.min(frame)
  labelim, nlabels = ndimage.label(mask)
  # each pixel in labelim contains labels of the object it belongs to.
  rects = []
  for labels in range(nlabels):
    clabel = labels+1
    o = ndimage.find_objects(labelim==clabel)
    # to get a bounding box
    # compute the (x, y)-coordinates of the bounding box for the object
    startX = o[0][0].start
    startY = o[0][1].start
    endX = o[0][0].stop
    endY = o[0][1].stop
    box = np.array([startX, startY, endX, endY])
    #print('box centroid is:',[int((startX + endX) / 2.0),int((startY + endY) / 2.0)])
    rects.append(box.astype("int"))
  return rects

def getObjectMotionDirection(objects, last_objects, rects, dims,FlowWidth):
  dirX = np.zeros(shape=(dims,dims))
  dirY = np.zeros(shape=(dims,dims))
  MotionAngles = np.zeros(shape=(dims,dims))
  objectIDs = list(objects.keys())
  objectCentroids = list(objects.values())
  last_objectIDs = list(last_objects.keys())
  last_objectCentroids = list(last_objects.values())
  directions = []
  locations = []
  for cvalue in objectIDs:
    cid = objectIDs.index(cvalue)
    cobj_centroid = objectCentroids[cid]
    if cvalue in last_objectIDs:
      lid = last_objectIDs.index(cvalue)
      lobj_centroid = last_objectCentroids[lid]
      for i in range(np.shape(rects)[0]):
        startX = rects[i][0]
        if startX<(FlowWidth/2):
          startX =  0
        else:
          startX = startX-(FlowWidth/2) 
        startY = rects[i][1]
        if startY<(FlowWidth/2):
          startY = 0
        else:
          startY = startY-(FlowWidth/2)
        endX = rects[i][2]
        if endX>dims-(FlowWidth/2):
          endX = dims
        else:
          endX = endX+(FlowWidth/2)
        endY = rects[i][3]
        if endY>dims-(FlowWidth/2):
          endY = dims
        else:
          endY = endY+(FlowWidth/2)
        if cobj_centroid[1]>=startY and cobj_centroid[1]<=endY and cobj_centroid[0]>=startX and cobj_centroid[0]<=endX:
          targetX = range(int(startX),int(endX),1)
          targetY = range(int(startY),int(endY),1)
      for ix in targetX:
        for iy in targetY:
          dirX[ix][iy]= cobj_centroid[1]-lobj_centroid[1] #x direction
          dirY[ix][iy]= cobj_centroid[0]-lobj_centroid[0] #y direction
      cdir = [cobj_centroid[1]-lobj_centroid[1],cobj_centroid[0]-lobj_centroid[0]]
      directions.append(cdir)
      locations.append([cobj_centroid[1],cobj_centroid[0]])
    else:
      lobj_centroid = []
  return dirX, dirY

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
NB_steps = 100
steps = 0
fig = plt.figure()
gs = gridspec.GridSpec(1,2)
f_ax = []
f_ax.append(fig.add_subplot(gs[0,0]))
f_ax.append(fig.add_subplot(gs[0,1]))
objects = OrderedDict()
last_objects = OrderedDict()
while steps<NB_steps:
  caction = random.randint(3,4)
  # read the next frame from the AIGame
  rewards, epCount, proposed_actions, total_hits, Racket_pos, Ball_pos = AIGame.playGame(actions=[caction], epCount = 0)
  frame = AIGame.FullImages[-1]
  #frame  = downscale_local_mean(frame,(8,8))
  # Detect the objects, and initialize the list of bounding box rectangles
  rects = getObjectsBoundingBoxes(frame)
  frame = np.ascontiguousarray(frame, dtype=np.uint8)
  # loop over rects
  for i in range(np.shape(rects)[0]):
    startX = rects[i][0]
    startY = rects[i][1]
    endX = rects[i][2]
    endY = rects[i][3]
    cv2.rectangle(frame, (startY, startX), (endY, endX),(0, 255, 0), 1)
  # update our centroid tracker using the computed set of bounding box rectangles
  objects = ct.update(rects)
  dirX, dirY = getObjectMotionDirection(objects, last_objects, rects, dims=160, FlowWidth=8)
  #dirX_ds = downscale_local_mean(dirX,(8,8))
  #dirY_ds = downscale_local_mean(dirY,(8,8))
  dirX_ds = resize(dirX,(20,20),anti_aliasing=True)
  dirY_ds = resize(dirY,(20,20),anti_aliasing=True)
  mag, ang = cv2.cartToPolar(dirX_ds, -1*dirY_ds)
  #mag, ang = cv2.cartToPolar(dirX, dirY)
  ang = np.rad2deg(ang)
  print(ang)
  last_objects = copy.deepcopy(objects)
  # loop over the tracked objects
  for (objectID, centroid) in objects.items():
    cv2.circle(frame, (centroid[1], centroid[0]), 1, (0, 255, 0), -1)
  if steps==0:
    im0 = f_ax[0].imshow(frame, origin='upper')
    X, Y = np.meshgrid(np.arange(0, 20, 1), np.arange(0,20,1))
    im1 = f_ax[1].quiver(X,Y,dirX_ds,-1*dirY_ds, pivot='mid', units='inches',width=0.022,scale=1/0.15)
    f_ax[1].set_xlim(0,20,1); f_ax[1].set_ylim(20,0,-1)
    plt.draw()
    plt.pause(1)
  else:
    im0.set_data(frame)
    im1.set_UVC(dirX_ds,-1*dirY_ds)
    plt.draw()
    plt.pause(1)
  last_object = objects
  steps = steps+1

