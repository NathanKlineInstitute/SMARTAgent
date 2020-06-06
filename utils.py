import os

def safemkdir (dn):
  # make a directory (dn), catch any exceptions; return True/False on success/failure
  try:
    os.mkdir(dn)
    return True
  except OSError:
    if not os.path.exists(dn):
      print('could not create', dn)
      return False
    else:
      return True

def backupcfg (name):
  # backup the config file to backupcfg subdirectory
  safemkdir('backupcfg')
  from conf import fnjson  
  fout = 'backupcfg/' + name + 'sim.json'
  if os.path.exists(fout):
    print('removing prior cfg file' , fout)
    os.system('rm ' + fout)
  os.system('cp ' + fnjson + '  ' + fout) # fcfg created in geom.py via conf.py

def getdatestr ():
  # get a date string (year,month,day)
  import datetime
  now = datetime.datetime.now()
  ttup = now.timetuple()
  syr = str(ttup.tm_year)[-2]+str(ttup.tm_year)[-1]
  dmon = {1:'jan',2:'feb',3:'mar',4:'apr',5:'may',6:'jun',7:'jul',8:'aug',9:'sep',10:'oct',11:'nov',12:'dec'}
  return syr + dmon[ttup.tm_mon] + str(ttup.tm_mday) + '_'

