# image processing utils
import numpy as np
import cv2 # opencv
# from skimage.registration import optical_flow_tvl1

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
  mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
  goodInds = np.where(mag<1e-10,0,1)
  return {'flow':flow,'mag':mag,'ang':np.rad2deg(ang),'goodInds':goodInds}

def getoptflowframes (Images,winsz=3, pyrscale=0.5, nlayer=3, niter=3, polyn=5, polysigma=1.1):
  # get optical flow between all frames in 3D array of frames; index 0 is frame; next indices are y,x
  return [getoptflow(Images[i,:,:],Images[i+1,:,:],winsz=winsz,pyrscale=pyrscale,nlayer=nlayer,niter=niter,polyn=polyn,polysigma=polysigma) for i in range(Images.shape[0]-1)]
