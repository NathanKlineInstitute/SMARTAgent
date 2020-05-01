# image processing utils
import numpy as np
import cv2 # opencv
# from skimage.registration import optical_flow_tvl1

def getoptflow (gimg0, gimg1, winsz=10, pyrscale=0.5, nlayer=3, niter=3, polyn=7, polysigma=1.5):
  # gets dense optical flow between two grayscale images (gimg0, gimg1)
  # using openCV's implementation the Gunnar Farneback's algorithm.
  # win size. flow is computed over the window. larger value is more robust to noise.
  # image pyramid or simple image scale
  # num opyramid layers. if use 1 means flow is calculated only from previous image.
  # num iterations
  # polynominal degree expansion (recommended 5-7)
  # standard deviation used to smooth used derivatives. (recommended 1.1-1.5)
  # see help(cv2.calcOpticalFlowFarneback) for param choices
  flow = cv2.calcOpticalFlowFarneback(gimg0,gimg1, None, pyrscale, nlayer, winsz, niter, polyn, polysigma, 0)
  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
  goodInds = np.where(mag<1e-10,0,1)
  return {'flow':flow,'mag':mag,'ang':ang,'goodInds':goodInds}

def getoptflowframes (Images):
  # get optical flow between all frames in 3D array of frames; index 0 is frame; next indices are y,x
  return [getoptflow(Images[i,:,:],Images[i+1,:,:]) for i in range(Images.shape[0]-1)]
