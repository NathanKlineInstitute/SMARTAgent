import os

# make dir, catch exceptions
def safemkdir (dn):
  try:
    os.mkdir(dn)
    return True
  except OSError:
    if not os.path.exists(dn):
      print('could not create', dn)
      return False
    else:
      return True

# backup the config file
def backupcfg (name):
  safemkdir('backupcfg')
  fout = 'backupcfg/' + name + 'sim.json'
  if os.path.exists(fout):
    print('removing prior cfg file' , fout)
    os.system('rm ' + fout)  
  os.system('cp sim.json ' + fout) # fcfg created in geom.py via conf.py
