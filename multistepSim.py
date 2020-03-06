import os
import json
import pickle
import sys

if __name__ == '__main__':
  basefn = sys.argv[1]
  ncore = int(sys.argv[2])
  nstep = int(sys.argv[3])
  outf = sys.argv[4]
  print(basefn,nstep)
  fpout = open(outf,'w')
  for i in range(nstep):
    d = json.load(open(basefn,'r'))
    simstr = d['sim']['name']
    d['sim']['name'] += '_step_' + str(i) + '_'
    if i > 0:
      d['simtype']['ResumeSim'] = 1
      d['simtype']['ResumeSimFromFile'] = 'data/' + simstr + '_step_' + str(i-1) + '_simConfig.pkl'
    fnjson = d['sim']['name'] + '.json'
    fpout.writelines('myrun ' + str(ncore) + ' ' + fnjson + '\n')
    json.dump(d, open(fnjson,'w'))
  fpout.close()
