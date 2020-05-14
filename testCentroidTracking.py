from centroidtracker import CentroidTracker
import numpy as np
import cv2
import random
from matplotlib import pyplot as plt
from scipy import ndimage
from aigame import AIGame
import time
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
    print('box centroid is:',[int((startX + endX) / 2.0),int((startY + endY) / 2.0)])
    rects.append(box.astype("int"))
  return rects

# initialize our centroid tracker and frame dimensions
ct = CentroidTracker()
objects = []
last_object = []
while True:
  caction = random.randint(3,4)
  # read the next frame from the AIGame
  rewards, epCount, proposed_actions, total_hits, Racket_pos, Ball_pos = AIGame.playGame(actions=[caction], epCount = 0)
  frame = AIGame.FullImages[-1]
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
    # if the object didn't move last time, delete it.
    #if len(objects)>0:
    #  cX = int((startX + endX) / 2.0)
    #  cY = int((startY + endY) / 2.0)
    #  newCentroid = [cX, cY]
    #  for (objectID, centroid) in objects.items():
    #    oldCentroid = centroid
    #    if (oldCentroid[0]-newCentroid[0]==0) and (oldCentroid[1]-newCentroid[1]==0):
    #      ct.deregister(objectID)
    #      ct.update([]) 
  # update our centroid tracker using the computed set of bounding box rectangles
  
  objects = ct.update(rects)
  # loop over the tracked objects
  for (objectID, centroid) in objects.items():
    # draw both the ID of the object and the centroid of the object on the output frame
    text = "ID {}".format(objectID)
    cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.circle(frame, (centroid[1], centroid[0]), 4, (0, 255, 0), -1)
  # show the output frame
  cv2.imshow("Frame", frame)
  time.sleep(10)
  last_object = objects
  key = cv2.waitKey(1) & 0xFF
  # if the `q` key was pressed, break from the loop
  if key == ord("q"):
    break
# do a bit of cleanup
cv2.destroyAllWindows()
