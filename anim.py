# utilities for animation output
import imageio
import os

# save set of images (specified by path) to output animated gif
def savegif (lfnimage, outpath):
  limage = [imageio.imread(fn) for fn in lfnimage]
  imageio.mimwrite(outpath, limage)

# save set of images specified by inpath to output mp4; framerate is frames per second
# framerate should be >= 10; otherwise ffmpeg may produce buggy output video
def savemp4 (inpath, outpath, framerate):
  import ffmpeg # pip install ffmpeg
  if os.path.isfile(outpath): os.unlink(outpath) # make sure can write to this path  
  (
    ffmpeg
    .input(inpath, pattern_type='glob', framerate=framerate)
    .output(outpath)
    .run()
  )
  
