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

def getObjectMotionDirection(objects, last_objects, rects, dims):
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
        startY = rects[i][1]
        endX = rects[i][2]
        endY = rects[i][3]
        if cobj_centroid[1]>=startY and cobj_centroid[1]<=endY and cobj_centroid[0]>=startX and cobj_centroid[0]<=endX:
          targetX = range(startX,endX,1)
          targetY = range(startY,endY,1)
      for ix in targetX:
        for iy in targetY:
          dirX[ix][iy]= cobj_centroid[1]-lobj_centroid[1] #x direction
          dirY[ix][iy]= cobj_centroid[0]-lobj_centroid[0] #y direction
      cdir = [cobj_centroid[1]-lobj_centroid[1],cobj_centroid[0]-lobj_centroid[0]]
      directions.append(cdir)
      locations.append([cobj_centroid[1],cobj_centroid[0]])
    else:
      lobj_centroid = []
  return dirX, -1*dirY


"""
# get oscillatory events
# lms is list of windowed morlet spectrograms (images), lmsnorm is spectrograms normalized by median in each power
# lnoise is whether the window had noise, medthresh is median threshold for significant events,
# lsidx,leidx are starting/ending indices into original time-series, csd is current source density
# on the single chan, MUA is multi-channel multiunit activity, overlapth is threshold for merging
# events when bounding boxes overlap, fctr is fraction of event amplitude to search left/right/up/down
# when terminating events
def getspecevents (lms,lmsnorm,lnoise,medthresh,lsidx,leidx,csd,MUA,chan,sampr,overlapth=0.5,fctr=0.5,getphase=False):
  llevent = []
  for windowidx,offidx,ms,msn,noise in zip(arange(len(lms)),lsidx,lms,lmsnorm,lnoise): 
    imgpk = detectpeaks(msn) # detect the 2D local maxima
    lblob = getblobsfrompeaks(msn,imgpk,ms.TFR,medthresh,fctr=fctr,T=ms.t,F=ms.f) # cut out the blobs/events
    lblobsig = [blob for blob in lblob if blob.maxval >= medthresh] # take only significant events
    lmergeset,bmerged = getmergesets(lblobsig,overlapth) # determine overlapping events
    lmergedblobs = getmergedblobs(lblobsig,lmergeset,bmerged)
    # get the extra features (before/during/after with MUA,avg,etc.)
    getextrafeatures(lmergedblobs,ms,msn,medthresh,csd,MUA,chan,offidx,sampr,fctr=fctr,getphase=getphase)
    for blob in lmergedblobs: # store offsets for getting to time-series / wavelet spectrograms
      blob.windowidx = windowidx
      blob.offidx = offidx
      blob.duringnoise = noise
    llevent.append(lmergedblobs) # save merged events
  return llevent

# finds boundaries where the image dips below the threshold, starting from x,y and moving left,right,up,down
def findbounds (img,x,y,thresh):
  ysz, xsz = img.shape
  y0 = y
  x0 = x - 1
  # look left
  while True:
    if x0 < 0:
      x0 = 0
      break
    if img[y0][x0] < thresh: break
    x0 -= 1
  left = x0
  # look right
  x0 = x + 1
  while True:
    if x0 >= xsz:
      x0 = xsz - 1
      break
    if img[y0][x0] < thresh: break
    x0 += 1
  right = x0
  # look down
  x0 = x
  y0 = y - 1
  while True:
    if y0 < 0:
      y0 = 0
      break
    if img[y0][x0] < thresh: break
    y0 -= 1
  bottom = y0
  # look up
  x0 = x
  y0 = y + 1
  while True:
    if y0 >= ysz:
      y0 = ysz - 1
      break
    if img[y0][x0] < thresh: break      
    y0 += 1
  top = y0
  #print('left,right,top,bottom:',left,right,top,bottom)  
  return left,right,top,bottom

# extract the event blobs from local maxima image (impk)
def getblobsfrompeaks (imnorm,impk,imorig,medthresh,fctr=0.5,T=None,F=None):
  # imnorm is normalized image, lbl is label image obtained from imnorm, imorig is original un-normalized image
  # medthresh is median threshold for significant peaks
  # getblobfeatures returns features of blobs in lbl using imnorm
  lpky,lpkx = np.where(impk) # get the peak coordinates
  lblob = []
  for y,x in zip(lpky,lpkx):
    pkval = imnorm[y][x]
    thresh = min(medthresh, fctr * pkval) # lower value threshold used to find end of event 
    left,right,top,bottom = findbounds(imnorm,x,y,thresh)
    #subimg = imnorm[bottom:top+1,left:right+1]
    #thsubimg = subimg > thresh
    #print('L,R,T,B:',left,right,top,bottom,subimg.shape,thsubimg.shape,sum(thsubimg))
    #print('sum(thsubimg)',sum(thsubimg),'amax(subimg)',amax(subimg))    
    b = evblob()
    #b.avgpoworig = ndimage.mean(imorig[bottom:top+1,left:right+1],thsubimg,[1])
    b.maxvalorig = imorig[y][x]
    #b.avgpow = ndimage.mean(subimg,thsubimg,[1])
    b.maxval = pkval
    b.minval = amin(imnorm[bottom:top+1,left:right+1])
    b.left = left
    b.right = right
    b.top = top
    b.bottom = bottom
    b.maxpos = (y,x)
    if F is not None:
      b.minF = F[b.bottom] # get the frequencies
      b.maxF = F[min(b.top,len(F)-1)]
      b.peakF = F[b.maxpos[0]]
      b.band = getband(b.peakF)
    if T is not None:
      b.minT = T[b.left]
      b.maxT = T[min(b.right,len(T)-1)]
      b.peakT = T[b.maxpos[1]]
    lblob.append(b)
  return lblob

#
def getmergesets (lblob,prct):
  #get the merged blobs (bounding boxes)
  #lblob is a list of blos (input)
  #prct is the threshold for fraction of overlap required to merge two blobs (boxes)
  #returns a list of sets of merged blobs and a bool list of whether each original blob was merged
  sz = len(lblob)
  bmerged = [False for i in range(sz)]
  for i,blob in enumerate(lblob): blob.ID = i # make sure ID assigned
  lmergeset = [] # set of merged blobs (boxes)
  for i in range(sz):
    blob0 = lblob[i]
    for j in range(sz):
      if i == j: continue
      blob1 = lblob[j]
      if blob0.getintersection(blob1).area() >= prct * min(blob0.area(),blob1.area()): # enough overlap between bboxes?
        # merge them
        bmerged[i]=bmerged[j]=True
        found = False
        for k,mergeset in enumerate(lmergeset): # determine if either of these bboxes are in existing mergesets
          if i in mergeset or j in mergeset: # one of the bboxes in an existing mergeset?
            found = True
            if i not in mergeset: mergeset.add(i) # i not already there? add it in
            if j not in mergeset: mergeset.add(j) # j not already there? add it in
        if not found: # did not find either bbox in an existing mergeset? then create a new mergeset
          mergeset = set()
          mergeset.add(i)
          mergeset.add(j)
          lmergeset.append(mergeset)
  return lmergeset, bmerged

#
def getmergedblobs (lblob,lmergeset,bmerged):
  # create a new list of blobs (boxes) based on lmergeset, and update the new blobs' properties
  lblobnew = [] # list of new blobs
  for i,blob in enumerate(lblob):
    if not bmerged[i]: lblobnew.append(blob) # non-merged blobs are copied as is
  for mergeset in lmergeset: # now go through the list of mergesets and create the new blobs
    lblobtmp = [lblob[ID] for ID in mergeset]
    for i,blob in enumerate(lblobtmp):
      if i == 0:
        box = bbox(blob.left,blob.right,blob.bottom,blob.top)
        peakF = blob.peakF
        minF = blob.minF
        maxF = blob.maxF
        minT = blob.minT
        maxT = blob.maxT
        peakT = blob.peakT
        maxpos = blob.maxpos
        maxval = blob.maxval
        minval = blob.minval
      else:
        box = box.getunion(blob)
        minF = min(minF, blob.minF)
        maxF = max(maxF, blob.maxF)
        minT = min(minT, blob.minT)
        maxT = max(maxT, blob.maxT)
        if blob.maxval > maxval:
          peakF = blob.peakF
          peakT = blob.peakT
          maxpos = blob.maxpos
          maxval = blob.maxval
        if blob.minval < minval:
          minval = blob.minval
    blob.left,blob.right,blob.bottom,blob.top = box.left,box.right,box.bottom,box.top
    blob.minF,blob.maxF,blob.peakF,blob.minT,blob.maxT,blob.peakT=minF,maxF,peakF,minT,maxT,peakT
    blob.maxpos,blob.maxval = maxpos,maxval
    blob.minval = minval
    lblobnew.append(blob)
  return lblobnew


"""
