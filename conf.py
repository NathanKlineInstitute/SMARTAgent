import json
import sys

fnjson = 'sim.json'

for i in range(len(sys.argv)):
  if sys.argv[i].endswith('.json'):
    fnjson = sys.argv[i]
    print('reading ', fnjson)

with open(fnjson,'r') as fp:
  dconf = json.load(fp)
  if 'DirectionDetectionAlgo' not in dconf:
    dconf['DirectionDetectionAlgo'] = {'CentroidTracker':0, 'OpticFlow':1}
  #print(dconf)

