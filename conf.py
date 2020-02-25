import json
import sys

fn = 'sim.json'
with open(fn,'r') as fp:
  dconf = json.load(fp)
  print(dconf)

