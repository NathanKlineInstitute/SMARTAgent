# utilities for animation output
import imageio
import os
from matplotlib import animation

# save set of images (specified by path) to output animated gif
def savegif (lfnimage, outpath):
  limage = [imageio.imread(fn) for fn in lfnimage]
  imageio.mimwrite(outpath, limage)

# save set of images specified by inpath to output mp4; framerate is frames per second
# framerate should be >= 10; otherwise ffmpeg may produce buggy output video
def savemp4 (inpath, outpath, framerate):
  import ffmpeg # pip install ffmpeg-python
  if os.path.isfile(outpath): os.unlink(outpath) # make sure can write to this path  
  (
    ffmpeg
    .input(inpath, pattern_type='glob', framerate=framerate)
    .output(outpath)
    .run()
  )
  
# ffmpeg.concat(split0, split1).output('out.mp4').run()

def getwriter (outpath, framerate):
  if outpath.endswith('mp4'):
    # Either avconv or ffmpeg need to be installed in the system to produce the videos!
    try:
      writer = animation.writers['ffmpeg']
    except KeyError:
      writer = animation.writers['avconv']
    writer = writer(fps=framerate)
    return writer
    ani.save(outpath, writer=writer)
  elif outpath.endswith('gif'):
    try:
      writer = animation.writers['imagemagick'] # need to have imagemagick installed
      writer = writer(fps=framerate)
      return writer
      ani.save(outpath, writer=writer)
    except:
      print('imagemagick not available')
  return None
  
