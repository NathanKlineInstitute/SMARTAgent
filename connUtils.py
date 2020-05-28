# neuronal network connection functions
import numpy as np

#
def gid2pos (numc, startgid, gid):
  nrow = ncol = int(np.sqrt(numc))
  y = int((gid - startgid) / nrow)
  x = (gid - startgid) % ncol
  return (x,y)

def prob2conv (prob, npre):
  # probability to convergence; prob is connection probability, npre is number of presynaptic neurons
  return int(0.5 + prob * npre)

def connectOnePreNtoOneMNeuron (NBNeurons,offset_pre,offset_post):
  #this method is used to generate list of connections between preSynNeurons and motor neurons.
  blist = []
  for i in range(NBNeurons):
    preN = i+offset_pre
    postN = i+offset_post
    blist.append([preN,postN])
  return blist
  
def connectLayerswithOverlap (NBpreN, NBpostN, overlap_xdir):
    #NBpreN = 6400 	#number of presynaptic neurons
    NBpreN_x = int(np.sqrt(NBpreN))
    NBpreN_y = int(np.sqrt(NBpreN))
    #NBpostN = 6400	#number of postsynaptic neurons
    NBpostN_x = int(np.sqrt(NBpostN))
    NBpostN_y = int(np.sqrt(NBpostN))
    convergence_factor = NBpreN/NBpostN
    convergence_factor_x = np.ceil(np.sqrt(convergence_factor))
    convergence_factor_y = np.ceil(np.sqrt(convergence_factor))
    #overlap_xdir = 5	#number of rows in a window for overlapping connectivity
    #overlap_ydir = 5	#number of columns in a window for overlapping connectivity
    overlap_ydir = overlap_xdir
    preNIndices = np.zeros((NBpreN_x,NBpreN_y))
    postNIndices = np.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
    blist = []
    for i in range(NBpreN_x):
        for j in range(NBpreN_y):
            preNIndices[i,j]=j+(NBpreN_y*i)
    for i in range(NBpostN_x):
        for j in range(NBpostN_y):
            postNIndices[i,j]=j+(NBpostN_y*i)
    for i in range(NBpostN_x):				#boundary conditions are implemented here
        for j in range(NBpostN_y):
            postN = int(postNIndices[i,j])
            if convergence_factor_x>1:
                preN = preNIndices[int(i*convergence_factor_y),int(j*convergence_factor_x)]
                #preN = int(convergence_factor_x*convergence_factor_y*NBpostN_y*i) + int(convergence_factor_y*j)
            else:
                preN = int(postN)
            preN_ind = np.where(preNIndices==preN)
            #print(preN_ind)
            x0 = preN_ind[0][0] - int(overlap_xdir/2)
            if x0<0:
                x0 = 0
            y0 = preN_ind[1][0] - int(overlap_ydir/2)
            if y0<0:
                y0 = 0
            xlast = preN_ind[0][0] + int(overlap_xdir/2)
            if xlast>NBpreN_x-1:
                xlast = NBpreN_x-1
            ylast = preN_ind[1][0] + int(overlap_ydir/2)
            if ylast>NBpreN_y-1:
                ylast = NBpreN_y-1
            xinds = [x0]
            for _ in range(xlast-x0):
                xinds.append(xinds[-1]+1)
            yinds = [y0]
            for _ in range(ylast-y0):
                yinds.append(yinds[-1]+1)
            for xi in range(len(xinds)):
                for yi in range(len(yinds)):
                    preN = int(preNIndices[xinds[xi],yinds[yi]])
                    blist.append([preN,postN]) 			#list of [presynaptic_neuron, postsynaptic_neuron] 
    return blist

def connectLayerswithOverlapDiv(NBpreN, NBpostN, overlap_xdir):
    NBpreN_x = int(np.sqrt(NBpreN))
    NBpreN_y = int(np.sqrt(NBpreN))
    NBpostN_x = int(np.sqrt(NBpostN))
    NBpostN_y = int(np.sqrt(NBpostN))
    divergence_factor = NBpostN/NBpreN
    divergence_factor_x = np.ceil(np.sqrt(divergence_factor))
    divergence_factor_y = np.ceil(np.sqrt(divergence_factor))
    overlap_ydir = overlap_xdir
    preNIndices = np.zeros((NBpreN_x,NBpreN_y))
    postNIndices = np.zeros((NBpostN_x,NBpostN_y))		#list created for indices from linear (1-6400) to square indexing (1-80,81-160,....) 
    blist = []
    for i in range(NBpreN_x):
        for j in range(NBpreN_y):
            preNIndices[i,j]=j+(NBpreN_y*i)
    for i in range(NBpostN_x):
        for j in range(NBpostN_y):
            postNIndices[i,j]=j+(NBpostN_y*i)
    for i in range(NBpreN_x):				#boundary conditions are implemented here
        for j in range(NBpreN_y):
            preN = int(preNIndices[i,j])
            if divergence_factor_x>1:
                postN = postNIndices[int(i*divergence_factor_y),int(j*divergence_factor_x)]
            else:
                postN = int(preN)
            postN_ind = np.where(postNIndices==postN)
            x0 = postN_ind[0][0] - int(overlap_xdir/2)
            if x0<0:
                x0 = 0
            y0 = postN_ind[1][0] - int(overlap_ydir/2)
            if y0<0:
                y0 = 0
            xlast = postN_ind[0][0] + int(overlap_xdir/2)
            if xlast>NBpostN_x-1:
                xlast = NBpostN_x-1
            ylast = postN_ind[1][0] + int(overlap_ydir/2)
            if ylast>NBpostN_y-1:
                ylast = NBpostN_y-1
            xinds = [x0]
            for _ in range(xlast-x0):
                xinds.append(xinds[-1]+1)
            yinds = [y0]
            for _ in range(ylast-y0):
                yinds.append(yinds[-1]+1)
            for xi in range(len(xinds)):
                for yi in range(len(yinds)):
                    postN = int(postNIndices[xinds[xi],yinds[yi]])
                    blist.append([preN,postN]) 			#list of [presynaptic_neuron, postsynaptic_neuron] 
    return blist
