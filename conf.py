import json
import sys
import getpass

fnjson = 'sim.json'

if getpass.getuser().count('samn') > 0: fnjson = 'sn.json' # this is just temporary - for samn

for i in range(len(sys.argv)):
  if sys.argv[i].endswith('.json'):
    fnjson = sys.argv[i]
    print('reading ', fnjson)

def readconf (fnjson):    
  with open(fnjson,'r') as fp:
    dconf = json.load(fp)
    #print(dconf)
  return dconf


def checkDefVal (d, k, val):
  # check if k is in d, if not, set d[k] = val
  if k not in d: d[k] = val

def ensureDefaults (dconf):
  # make sure (some of the) default values are present so dont have to check for them throughout rest of code
  if 'DirectionDetectionAlgo' not in dconf:
    dconf['DirectionDetectionAlgo'] = {'CentroidTracker':0, 'OpticFlow':1, 'UseFull':1}  
  checkDefVal(dconf,'net',{})
  checkDefVal(dconf['net'], 'EEPreMProb', 0.0)  
  checkDefVal(dconf['net'], 'EEMProb', 0.1)
  checkDefVal(dconf['net'], 'EEMRecProb', 0.0)
  checkDefVal(dconf['net'], 'EEMFeedbackProb', 0.0)  
  checkDefVal(dconf, 'verbose', 0)
  checkDefVal(dconf['net'], 'ECellModel', 'Mainen')
  checkDefVal(dconf['net'], 'ICellModel', 'FS_BasketCell')
  checkDefVal(dconf['net'], 'delayMinSoma', 1.8)
  checkDefVal(dconf['net'], 'delayMaxSoma', 2.2)    
  checkDefVal(dconf['net'], 'delayMinSTIMMOD', 1.8)
  checkDefVal(dconf['net'], 'delayMaxSTIMMOD', 2.2)
  checkDefVal(dconf['net'], 'delayMinDend', 3)
  checkDefVal(dconf['net'], 'delayMaxDend', 10)      
  for k in ['VisualRL', 'EIPlast', 'VisualFeedback', 'useNeuronPad']: checkDefVal(dconf['net'], k, False)
  for k in ['EEGain', 'EIGain', 'IEGain', 'IIGain', 'scale']: checkDefVal(dconf['net'], k, 1.0)
  for k in ['useBinaryImage', 'EXPDir', 'VTopoI']: checkDefVal(dconf['net'],k,True)
  checkDefVal(dconf['net'], 'DirMinRate', 0.0)
  checkDefVal(dconf['net'], 'DirMaxRate', 150.0)
  checkDefVal(dconf['net'], 'LocMaxRate', 150.0)  
  checkDefVal(dconf['net'], 'FiringRateCutoff', 50.0)
  checkDefVal(dconf['net'], 'stimModDirW', 0.02)
  checkDefVal(dconf['net'], 'stimModInputW', 0.02)
  checkDefVal(dconf['net'], 'weightVar', 0.0)
  checkDefVal(dconf, 'movefctr', 1.0)
  checkDefVal(dconf,"0rand",0)
  checkDefVal(dconf,"randmove",0)
  checkDefVal(dconf,"stochmove",0)
  checkDefVal(dconf,"avoidStuck",0)
  checkDefVal(dconf,"useImagePadding",0)    
  for k in ["actionsPerPlay", "followOnlyTowards", "useRacketPredictedPos"]: checkDefVal(dconf, k, 1)
  checkDefVal(dconf, 'stayStepLim', 0)
  for k in ['anticipatedRL', 'RLFakeUpRule', 'RLFakeDownRule', 'RLFakeStayRule', 'doplot', 'saveCellSecs', 'saveCellConns']:
    checkDefVal(dconf['sim'], k, 0)
  checkDefVal(dconf['sim'], 'saveMotionFields', 1)
  checkDefVal(dconf['sim'], 'targettedRL', 1)
  checkDefVal(dconf['sim'], 'targettedRLOppFctr', 0.5)
  checkDefVal(dconf['sim'], 'targettedRLDscntFctr', 0.5)
  if 'alltopoldivcons' not in dconf['net']:
    dconf['net']['alltopoldivcons'] = {
      "IR":{"ER":5},
      "EV1":{"ER":3},
      "IV1":{"EV1":5,"ER":5},
      "EV4":{"EV1":3,"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "IV4":{"EV4":5,"EV1":5},
      "EMT":{"EV4":3,"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "IMT":{"EMT":5,"EV4":5}      
    }
  if 'alltopolconvcons' not in dconf['net']:
    dconf['net']['alltopolconvcons'] = {
      "ER":{"IR":3,"EV1":3,"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1":{"IV1":3,"EV4":3,"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DE":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DNE":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DN":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DNW":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DW":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DSW":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DS":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
      "EV1DSE":{"EMDOWN":3,"EMUP":3,"EMSTAY":3},
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
    checkDefVal(dconf,'simulatedEnvParams',{})
    checkDefVal(dconf['simulatedEnvParams'], 'possible_ball_dy' : [1,1,1,1,1,1,2,2,2,2,3,3,3])
    checkDefVal(dconf['simulatedEnvParams'], 'possible_ball_dx' : [1,1,1,1,1,1,2,2,2,2,3,3,3])
    checkDefVal(dconf['simulatedEnvParams'], 'top_bottom_rule', 1)
    

dconf = readconf(fnjson) # read the configuration
ensureDefaults(dconf) # ensure some of the default values present

