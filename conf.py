import json
import sys

fnjson = 'sim.json'

for i in range(len(sys.argv)):
  if sys.argv[i].endswith('.json'):
    fnjson = sys.argv[i]
    print('reading ', fnjson)

def readconf (fnjson):    
  with open(fnjson,'r') as fp:
    dconf = json.load(fp)
    if 'DirectionDetectionAlgo' not in dconf:
      dconf['DirectionDetectionAlgo'] = {'CentroidTracker':0, 'OpticFlow':1, 'UseFull':1}
    #print(dconf)
  return dconf

def ensureDefaults (dconf):
  # make sure (some of the) default values are present so dont have to check for them throughout rest of code
  if 'verbose' not in dconf: dconf['verbose'] = 0
  if 'EcellModel' not in dconf['net']: dconf['net']['ECellModel'] = 'Mainen'
  if 'ICellModel' not in dconf['net']: dconf['net']['ICellModel'] = 'FS_BasketCell'
  for k in ['EEGain', 'EIGain', 'IEGain', 'IIGain', 'scale']:
    if k not in dconf['net']: dconf['net'][k] = 1.0
  for k in ['useBinaryImage', 'useNeuronPad', 'EXPDir']:
    if k not in dconf['net']: dconf['net'][k] = True
  if 'movefctr' not in dconf: dconf['movefctr'] = 1.0
  for k in ["actionsPerPlay", "followOnlyTowards", "useRacketPredictedPos"]:
    if k not in dconf: dconf[k] = 1
  if 'stayStepLim' not in dconf: dconf['stayStepLim'] = 0
  for k in ['anticipatedRL', 'RLFakeUpRule', 'RLFakeDownRule', 'RLFakeStayRule', 'doplot', 'saveCellSecs', 'saveCellConns']:
    if k not in dconf['sim']:
      dconf['sim'][k] = 0
  if 'alltopoldivcons' not in dconf['net']:
    dconf['net']['alltopoldivcons'] = {
      "IR":{"ER":5},
      "EV1":{"ER":3},
      "IV1":{"EV1":5,"ER":5},
      "EV4":{"EV1":3},
      "IV4":{"EV4":5,"EV1":5},
      "EMT":{"EV4":3},
      "IMT":{"EMT":5,"EV4":5}
    }
  if 'alltopolconvcons' not in dconf['net']:
    dconf['net']['alltopolconvcons'] = {
      "ER":{"IR":3,"EV1":3},
      "EV1":{"IV1":3,"EV4":3},
      "IV1":{"IV4":5},
      "EV4":{"IV4":3,"EMT":3},
      "IV4":{"IMT":5},
      "EMT":{"IMT":3}
    }
    if 'Noise' not in dconf:
      dconf['Noise'] = {
        "I": {
          "Rate": 0, 
          "Weight": 1e-05
        }, 
        "E": {
          "Rate": 0, 
          "Weight": 1e-05
        }
      }
    if 'architecturePreMtoM' not in dconf:
      dconf['architecturePreMtoM'] = {
        "useProbabilistic": 1, 
        "useTopological": 0
    }
    

dconf = readconf(fnjson) # read the configuration
ensureDefaults(dconf) # ensure some of the default values present

