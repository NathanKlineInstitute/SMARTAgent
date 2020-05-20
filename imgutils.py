# image processing utils
import numpy as np
import cv2 # opencv
# from skimage.registration import optical_flow_tvl1
from scipy import ndimage
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion  
from collections import OrderedDict
import copy
from skimage.transform import downscale_local_mean, rescale

def getoptflow (gimg0, gimg1, winsz=3, pyrscale=0.5, nlayer=3, niter=3, polyn=5, polysigma=1.1):
  # gets dense optical flow between two grayscale images (gimg0, gimg1)
  # using openCV's implementation the Gunnar Farneback's algorithm.
  """
    .   @param winsz averaging window size; larger values increase the algorithm robustness to image
    .   noise and give more chances for fast motion detection, but yield more blurred motion field.
    .   @param pyr_scale parameter, specifying the image scale (\<1) to build pyramids for each image;
    .   pyrscale=0.5 means a classical pyramid, where each next layer is twice smaller than the previous
    .   one.
    .   @param nlayer number of pyramid layers including the initial image; levels=1 means that no extra
    .   layers are created and only the original images are used.
    .   noise and give more chances for fast motion detection, but yield more blurred motion field.
    .   @param niter number of iterations the algorithm does at each pyramid level.
    .   @param polyn size of the pixel neighborhood used to find polynomial expansion in each pixel;
    .   larger values mean that the image will be approximated with smoother surfaces, yielding more
    .   robust algorithm and more blurred motion field, typically poly_n =5 or 7.
    .   @param polysigma standard deviation of the Gaussian that is used to smooth derivatives used as a
    .   basis for the polynomial expansion; for polyn=5, you can set polysigma=1.1, for polyn=7, a
    .   good value would be polysigma=1.5.
  """
  # see help(cv2.calcOpticalFlowFarneback) for param choices
  flow = cv2.calcOpticalFlowFarneback(gimg0,gimg1, None, pyrscale, nlayer, winsz, niter, polyn, polysigma, 0)
  mag, ang = cv2.cartToPolar(flow[...,0], -flow[...,1])
  ang = np.rad2deg(ang)
  thang = np.copy(ang) # now perform thresholding
  th = np.mean(mag) + np.std(mag)
  goodInds = np.where(mag<th,0,1)
  thflow = np.copy(flow)
  for y in range(thang.shape[0]):
    for x in range(thang.shape[1]):
      if mag[y,x] < th:
        thang[y,x] = -100 # invalid angle; angles should all be non-negative
        thflow[y,x,0] = thflow[y,x,1] = 0 # 0 flow
  return {'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds,'thang':thang,'thflow':thflow}

def getoptflowframes (Images,winsz=3, pyrscale=0.5, nlayer=3, niter=3, polyn=5, polysigma=1.1):
  # get optical flow between all frames in 3D array of frames; index 0 is frame; next indices are y,x
  return [getoptflow(Images[i,:,:],Images[i+1,:,:],winsz=winsz,pyrscale=pyrscale,nlayer=nlayer,niter=niter,polyn=polyn,polysigma=polysigma) for i in range(Images.shape[0]-1)]

# from https://stackoverflow.com/questions/3684484/peak-detection-in-a-2d-array
def detectpeaks (image):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """
  # define an 8-connected neighborhood
  neighborhood = generate_binary_structure(2,2)
  #apply the local maximum filter; all pixel of maximal value 
  #in their neighborhood are set to 1
  local_max = maximum_filter(image, footprint=neighborhood)==image
  #local_max is a mask that contains the peaks we are 
  #looking for, but also the background.
  #In order to isolate the peaks we must remove the background from the mask.
  #we create the mask of the background
  background = (image==0)
  #a little technicality: we must erode the background in order to 
  #successfully subtract it form local_max, otherwise a line will 
  #appear along the background border (artifact of the local maximum filter)
  eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
  #we obtain the final mask, containing only peaks, 
  #by removing the background from the local_max mask (xor operation)
  detected_peaks = local_max ^ eroded_background
  return detected_peaks

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

def getObjectMotionDirection(objects, last_objects, rects, dims, FlowWidth):
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
