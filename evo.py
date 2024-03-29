import sys
import random 
import time 
import inspyred
from inspyred import ec
from inspyred.ec import terminators
from inspyred.ec.variators import mutator
from inspyred.ec.generators import diversify
from random import Random
from time import time, clock
import datetime # to format time of run
import os
import tempfile
import pickle
import logging
from subprocess import Popen, PIPE, call
from neuron import h
import shlex
import copy
from functools import wraps
import numpy as np
import random
import os
import json
import pickle
import sys
import pandas as pd
from simdat import readweightsfile2pdf, pdf2weightsdict

pc = h.ParallelContext()
mydir = os.getcwd()
simf = mydir + '/sim.py' 
ngen = 0
nfunc = 0 # number of fitness functions used

# make dir, catch exceptions
def safemkdir (dn):
  if os.path.exists(dn): return True
  try:
    os.mkdir(dn)
    return True
  except OSError:
    if not os.path.exists(dn):
      print('could not create', dn)
      return False
    else:
      return True

# backup the config file, return path to backed-up file
def backupcfg (evostr,fcfg):
  fout = 'backupcfg/' + evostr + '/evo.json'
  if os.path.exists(fout):
    print('removing prior json file' , fout)
    os.system('rm ' + fout)  
  os.system('cp ' + fcfg + ' ' + fout) # fcfg created conf.py
  with open('backupcfg/' + evostr + '/command.txt','a') as fp: # write command line args too
    for s in sys.argv: fp.write(' ' + s)
    fp.write('\n')
  return fout 

#
def FitJobStrFN (p, args, cdx):
  global es,ncore # global evolution object
  pdfnew = args['startweight'].copy() # copy starting synaptic weights
  for i in range(len(p)): pdfnew.at[i,'weight'] = p[i] # update the weights based on the candidate
  # next update the simulation's json file
  simconfig = args['simconfig']
  d = json.load(open(simconfig,'r')) # original input json
  d['sim']['name'] += '_evo_gen_' + str(args['_ec'].num_generations) + '_cand_' + str(cdx) + '_' # also include candidate ID
  simstr = d['sim']['name']
  fnweight = mydir+'/evo/'+ simstr + 'weight.pkl' # filename for weights
  d['sim']['doquit'] = 1; d['sim']['doplot'] = 0
  d['simtype']['ResumeSim'] = 1 # make sure simulation loads the weights
  d['simtype']['ResumeSimFromFile'] = fnweight
  fnjson = mydir+'/evo/'+d['sim']['name'] + 'sim.json'
  json.dump(d, open(fnjson,'w'), indent=2)
  # save synaptic weights to file
  pdfnew = pdfnew[pdfnew.time==np.amax(pdfnew.time)]  
  D = pdf2weightsdict(pdfnew)
  pickle.dump(D, open(fnweight,'wb')) # temp file for synaptic weights
  # generate a command for running the simulation
  strc = './myrun ' + str(ncore) + ' ' + fnjson # command string  
  return strc,'data/'+d['sim']['name']+'ActionsRewards.txt'

# evaluate fitness with sim run
def EvalFIT (candidates, args):
  global es, quitaftermiss
  fitness = []; 
  for cdx, p in enumerate(candidates):
    strc,fn = FitJobStrFN(p, args, cdx)
    print('EvalFIT:', cdx, strc, fn)
    ret = os.system(strc)
    actreward = pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit','followtargetsign'])
    if quitaftermiss:
      fit = np.amax(actreward['time'])
    else:
      fit = np.sum(actreward['reward'])
    print('fit is:',fit)
    logger.info(strc + ', ' + fn + ', fit=' + str(fit))
    fitness.append(fit)
  return fitness

weightVar = 0.9
wmin = 1.0 - weightVar
wmax = 1.0 + weightVar

# generate params 
@diversify
def my_generate (random, args):
  pout = []
  W = args['startweight']['weight']
  pdf = args['startweight']
  for i in range(len(W)):
    pout.append(max(0,random.uniform(wmin * pdf.at[i,'weight'], wmax * pdf.at[i,'weight'])))
  return pout

# run sim command via mpi, then delete the temp file. returns job index and fitness.
def RunViaMPI (cdx, cmd, fn, maxfittime):
  global pc, useEMO, quitaftermiss
  if nfunc > 1 and useEMO: fit = [1e9 for i in range(nfunc)]
  else: fit = 1e9
  #print 'pc.id()==',pc.id(),'. starting py job', cdx, 'command:', cmd
  cmdargs = shlex.split(cmd)
  #proc = Popen(cmdargs)
  proc = Popen(cmdargs,stdout=PIPE,stderr=PIPE)
  #print 'py job', cdx, ' called popen'
  cstart = time(); killed = False
  while not killed and proc.poll() is None: # job is not done
    #print 'py job', cdx, ' called poll'
    cend = time(); rtime = cend - cstart
    if rtime >= maxfittime:
      killed = True
      print(' ran for ' , round(rtime,2) , 's. too slow , killing.')
      try:
        proc.kill() # has to be called before proc ends
        #proc.terminate()
      except:
        print('could not kill')
  #print 'py job', cdx, ' reading output'
  if not killed:
    #print 'not killed'
    try: proc.communicate() # avoids deadlock due to stdout/stderr buffer overfill
    except: print('could not communicate') # Process finished.
    try: # lack of output file may occur if invalid param values lead to an nrniv crash
      #print 'here'
      #print 'trying to read single fit value'
      # with open(fn,'r')  as fp: fit = float(fp.readlines()[0].strip())
      actreward = pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit','followtargetsign'])
      if quitaftermiss:
        fit = np.amax(actreward['time'])
      else:
        fit = np.sum(actreward['reward'])      
    except:
      print('WARN: could not read.')
  #print 'py job', cdx, ' removing temp file'
  os.unlink(fn)
  #print 'pc.id()==',pc.id(),'py job', cdx, ' returning', fit
  return cdx,fit

def printfin (fin):
  sys.stdout.write('\rFinished:{0}...'.format(fin));
  sys.stdout.flush()

# evaluate fitness with sim run
def EvalFITMPI (candidates, args):
  global pc
  #print 'pcid' , pc.id(), ' len(candidates)', len(candidates) # only pc.id() == 0 goes here
  fitness = [1e9 for i in range(len(candidates))]; 
  lcomm = ['' for i in range(len(candidates))] # commands
  lfile = ['' for i in range(len(candidates))] # temp files
  for cdx,p in enumerate(candidates): lcomm[cdx],lfile[cdx] = FitJobStrFN(p,args,cdx)
  # print lcomm[0]
  verbose=args['verbose']
  njob = len(candidates); cdx = 0
  maxfittime = args['maxfittime']
  for i in range(int(pc.nhost())): # first submit nhost jobs
    if cdx >= njob: break
    pc.submit(RunViaMPI, cdx,lcomm[cdx],lfile[cdx],maxfittime)
    cdx += 1
  finished = 0;
  if verbose: printfin(finished)
  while cdx < njob: # then submit the remainder
    if verbose: printfin(finished)
    while pc.working(): # submit a job each time one job finishes
      outidx,fit = pc.pyret()
      fitness[int(outidx)] = fit
      finished += 1; 
      if verbose: printfin(finished)
      if cdx < njob:
        pc.submit(RunViaMPI, cdx,lcomm[cdx],lfile[cdx],maxfittime)
        cdx += 1
  while pc.working(): # then wait for the remainder
    outidx,fit = pc.pyret()
    fitness[int(outidx)] = fit
    finished += 1;
    if verbose: printfin(finished)
  if verbose: print('')
  return fitness

#
def setEClog ():
  global logger
  logger = logging.getLogger('inspyred.ec')
  logger.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler(evostr+'.log', mode='w')
  file_handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger

# saves individuals in a population to binary file (pkl)
def my_indiv_observe (population, num_generations, num_evaluations, args):
  fn = 'data/' + evostr + '/gen_' + str(num_generations) + '_indiv.pkl'
  print('type(population=)',type(population))
  pickle.dump(population,open(fn,'wb'))

es = None
      
# use the archive_seeds to initialize an archive
def initialize_archive(f, archive_seeds):
  @wraps(f)
  def wrapper(random, population, archive, args):
    if args['_ec'].num_generations == 0:
      return archive_seeds
    else:
      return f(random, population, archive, args)
  return wrapper

# run the evolution
def runevo (popsize=100,maxgen=10,my_generate=my_generate,\
            nproc=16,ncore=8,rdmseed=1234,useDEA=True,\
            fstats='/dev/null',findiv='/dev/null',mutation_rate=0.2,useMPI=False,\
            numselected=100,noBound=False,simconfig='sn.json',\
            useLOG=False,maxfittime=600,lseed=None,larch=None,verbose=True,\
            useundefERR=False,startweight=None):
  global es
  useMProc = False
  if useLOG: logger=setEClog()
  rand = Random(); rand.seed(rdmseed) # alternative - provide int(time.time()) as seed
  if useDEA: # differential evolution
    es = ec.DEA(rand)
  else: # evolution strategy
    es = ec.ES(rand)
  es.terminator = terminators.generation_termination 
  #es.variator = [inspyred.ec.variators.heuristic_crossover,my_mutation]#inspyred.ec.variators.nonuniform_mutation
  # es.variator = [inspyred.ec.variators.heuristic_crossover] # ,inspyred.ec.variators.nonuniform_mutation]
  es.observer = [] # my_indiv_observe] # saves individuals to pkl file each generation
  statfile = open(fstats,'w'); indfile = open(findiv,'w')
  es.observer.append(inspyred.ec.observers.file_observer)
  es.observer.append(inspyred.ec.observers.stats_observer)
  # es.selector = inspyred.ec.selectors.tournament_selection
  # es.replacer = inspyred.ec.replacers.generational_replacement#inspyred.ec.replacers.plus_replacement
  # if noBound: es.observer.append(my_bound_observe)
  if useMPI:
    pc.barrier()
    if pc.id()==0: print('nhost : ' , pc.nhost())
    pc.runworker()
    print('before evolve my id is ' , pc.id())# checked that only host 0 calls evolution.
    final_pop = es.evolve(generator=my_generate,
                            evaluator=EvalFITMPI, 
                            pop_size=popsize,
                            maximize=True,
                            bounder=ec.Bounder(0,35.0),                            
                            max_generations=maxgen,
                            statistics_file=statfile,
                            individuals_file=indfile,
                            mutation_rate=mutation_rate,
                            simf=simf,
                            num_selected=numselected,
                            simconfig=simconfig,
                            noBound=noBound,
                            es=es,
                            maxfittime=maxfittime,
                            seeds=lseed,
                            verbose=verbose,
                            useundefERR=useundefERR,
                            startweight=startweight)
    print('after evolve my id is ' , pc.id()) # checked that only host 0 calls evolution.
    pc.done()
  elif useMProc: # use multiprocessing
    print('using multiprocessing')
    final_pop = es.evolve(generator=my_generate,
                            evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                            mp_evaluator=EvalFIT, 
                            mp_nprocs=nproc,
                            pop_size=popsize,
                            maximize=True,
                            bounder=ec.Bounder(0,35.0),                            
                            max_generations=maxgen,
                            statistics_file=statfile,
                            individuals_file=indfile,
                            mutation_rate=mutation_rate,
                            simf=simf,
                            num_selected=numselected,
                            simconfig=simconfig,
                            noBound=noBound,
                            es=es,
                            maxfittime=maxfittime,
                            seeds=lseed,
                            verbose=verbose,
                            useundefERR=useundefERR,
                            startweight=startweight)
  else:
    final_pop = es.evolve(generator=my_generate,
                            evaluator=EvalFIT,
                            pop_size=popsize,
                            maximize=True,
                            bounder=ec.Bounder(0,35.0),
                            max_generations=maxgen,
                            statistics_file=statfile,
                            individuals_file=indfile,
                            mutation_rate=mutation_rate,
                            simf=simf,
                            num_selected=numselected,
                            simconfig=simconfig,
                            noBound=noBound,
                            es=es,
                            maxfittime=maxfittime,
                            seeds=lseed,
                            verbose=verbose,
                            useundefERR=useundefERR,
                            startweight=startweight)
    
  # Sort and print the best individual
  final_pop.sort(reverse=False)
  statfile.close(); indfile.close()
  return [final_pop]

# plot generation progress over time
def genplot (fn='evo_stats.txt'): inspyred.ec.analysis.generation_plot(fn)

# individual pkl (or archive file) to list of candidates (each entry has params)
def indivpkl2list (fname):
  ark = pickle.load(open(fname))
  if type(ark[0])==list: lout = [indiv for indiv in ark]
  else: lout = [indiv.candidate for indiv in ark]
  return lout

if __name__ == "__main__":
  if len(sys.argv) < 2:
    print('usage: python evo.py [popsize n] [maxgen n] [nproc n] [useMPI 0/1] [useDEA 0/1] [mutation_rate 0--1] [numselected n] [evostr name] [simf name]')
    print('[noBound 0/1] [maxfittime seconds][simconfig path][verbose 0/1][rdmseed int]')
    quit()

  popsize=100; maxgen=10; nproc=16; useMPI=True; numselected=100; useDEA = False;
  mutation_rate=0.2; evostr='21aug4A'; simconfig = 'sn.json'; maxfittime = 600; 
  noBound = False; useLOG = True; useEMO = False
  verbose = True; rdmseed=1234; useundefERR = False; 
  fseed = farch = lseed = larch = None; # files,lists for initial population and archive
  fstat = findiv = '/dev/null'; quitaftermiss = True
  i = 1; narg = len(sys.argv)
  while i < narg:
    if sys.argv[i] == 'popsize' or sys.argv[i] == '-popsize':
      if i+1<narg:
        i+=1; popsize = int(sys.argv[i]); 
    elif sys.argv[i] == 'maxgen' or sys.argv[i] == '-maxgen':
      if i+1 < narg:
        i+=1; maxgen = int(sys.argv[i]); 
    elif sys.argv[i] == 'nproc' or sys.argv[i] == '-nproc':
      if i+1 < narg:
        i+=1; nproc = int(sys.argv[i]);
    elif sys.argv[i] == 'ncore' or sys.argv[i] == '-ncore':
      if i+1 < narg:
        i+=1; ncore = int(sys.argv[i]);    
    elif sys.argv[i] == 'rdmseed' or sys.argv[i] == '-rdmseed':
      if i+1 < narg:
        i+=1; rdmseed = int(sys.argv[i]); 
    elif sys.argv[i] == 'useMPI' or sys.argv[i] == '-useMPI':
      if i+1 < narg:
        i+=1; useMPI = bool(int(sys.argv[i])); 
    elif sys.argv[i] == 'useDEA' or sys.argv[i] == '-useDEA':
      if i+1 < narg:
        i+=1; useDEA = bool(int(sys.argv[i])); 
    elif sys.argv[i] == 'numselected' or sys.argv[i] == '-numselected':
      if i+1 < narg:
        i+=1; numselected = int(sys.argv[i]);
    elif sys.argv[i] == 'evostr' or sys.argv[i] == '-evostr':
      if i+1 < narg:
        i+=1; evostr = sys.argv[i]
    elif sys.argv[i] == 'simconfig' or sys.argv[i] == '-simconfig':
      if i+1 < narg:
        i+=1; simconfig = sys.argv[i]
    elif sys.argv[i] == 'mutation_rate' or sys.argv[i] == '-mutation_rate' or sys.argv[i] == '-mutationrate' or sys.argv[i] == 'mutationrate':
      if i+1 < narg:
        i+=1; mutation_rate = float(sys.argv[i])
    elif sys.argv[i] == 'useLOG' or sys.argv[i] == '-useLOG':
      if i+1 < narg:
        i+=1; useLOG = bool(int(sys.argv[i]));
    elif sys.argv[i] == 'verbose' or sys.argv[i] == '-verbose':
      if i+1 < narg:
        i+=1; verbose = bool(int(sys.argv[i]));
    elif sys.argv[i] == 'noBound' or sys.argv[i] == '-noBound':
      if i+1 < narg:
        i+=1; noBound = bool(int(sys.argv[i]));
    elif sys.argv[i] == 'useundefERR' or sys.argv[i] == '-useundefERR':
      if i+1 < narg:
        i+=1; useundefERR = bool(int(sys.argv[i]));
    elif sys.argv[i] == 'simf' or sys.argv[i] == '-simf':
      if i+1 < narg:
        i+=1; simf = sys.argv[i]
    elif sys.argv[i] == 'maxfittime' or sys.argv[i] == '-maxfittime':
      if i+1 < narg:
        i+=1; maxfittime = float(sys.argv[i])
    elif sys.argv[i] == 'fseed':
      if i+1 < narg:
        i+=1; fseed = sys.argv[i]; lseed = indivpkl2list(fseed)
    elif sys.argv[i] == 'farch':
      if i+1 < narg:
        i+=1; farch = sys.argv[i]; larch = pickle.load(open(farch))
    elif sys.argv[i] == 'startweight':
      if i+1 < narg:
        i+=1; startweight = readweightsfile2pdf(sys.argv[i]) # starting weights - placeholders
        # print('startweight columns:',startweight.columns)
    elif sys.argv[i] == 'fstats':
      if i+1 < narg:
        i+=1; fstats = sys.argv[i]
    elif sys.argv[i] == 'findiv':
      if i+1 < narg:
        i+=1; findiv = sys.argv[i]
    elif sys.argv[i] == 'quitaftermiss':
      if i+1 < narg:
        i+=1; quitaftermiss = bool(int(sys.argv[i]))
    elif sys.argv[i] == '-python' or sys.argv[i] == '-mpi' or sys.argv[i] == 'evo.py':
      pass
    else: raise Exception('unknown arg:'+sys.argv[i])
    i+=1;

  if (useMPI and pc.id()==0) or not useMPI:
    print('popsize:',popsize,'maxgen:',maxgen,'nproc:',nproc,'ncore:',ncore,'useMPI:',useMPI,'numselected:',numselected,'evostr:',evostr,\
          'useDEA:',useDEA,'mutation_rate:',mutation_rate,'noBound:',noBound,'useLOG:',useLOG,\
          'maxfittime:',maxfittime,'fseed:',fseed,'farch:',farch,'verbose:',verbose,'rdmseed:',rdmseed,'useundefERR:',useundefERR,'fstats:',fstats,'findiv:',findiv)
      
  # make sure master node does not work on submitted jobs (that would prevent it managing/submitting other jobs)
  if useMPI and pc.id()==0: pc.master_works_on_jobs(0) 

  if (useMPI and pc.id()==0) or not useMPI:
    # backup the config file and use backed-up version for evo (in case local version changed during evolution)
    safemkdir('backupcfg/'+evostr) # make a data output dir
    simconfig = backupcfg(evostr,simconfig) 
    safemkdir(mydir+'/evo') # for temp files

  myout = runevo(popsize=popsize,maxgen=maxgen,nproc=nproc,rdmseed=rdmseed,useMPI=useMPI,\
                 numselected=numselected,mutation_rate=mutation_rate,\
                 useDEA=useDEA,fstats=fstats,findiv=findiv,simconfig=simconfig,\
                 useLOG=useLOG,maxfittime=maxfittime,lseed=lseed,larch=larch,\
                 verbose=verbose,useundefERR=useundefERR,startweight=startweight)

  if (useMPI and pc.id()==0) or not useMPI:
    pickle.dump(myout[0],open('data/' + evostr + '/fpop.pkl','wb'))
    if useEMO: pickle.dump(myout[1],open('data/' + evostr + '/ARCH.pkl','wb'))

  if useMPI:
    if pc.id()==0: print('MPI finished, exiting.')
    quit() # make sure CPUs freed
  
