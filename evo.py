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
# import sim # runs different cell models (most contain Ben Suter ion channels)
import pickle
import logging
from subprocess import Popen, PIPE, call
# import paramiko # py SSH
# import popen2
from neuron import h
import shlex
# import lex
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
  fout = 'data/' + evostr + '/evo.cfg'
  if os.path.exists(fout):
    print('removing prior cfg file' , fout)
    os.system('rm ' + fout)  
  os.system('cp ' + fcfg + ' ' + fout) # fcfg created in geom.py via conf.py
  with open(fout,'a') as fp: # write command line args too
    fp.write('################\n')
    fp.write('# command-line args:\n')
    fp.write('#')
    for s in sys.argv: fp.write(' ' + s)
    fp.write('\n')
  return fout 

#
def FitJobStrFN (p, args, cdx):
  global es
  pdfnew = args['startweight'].copy() # copy starting synaptic weights
  print('pdfnew columns:',pdfnew.columns)
  #Wnew = pdfnew['weight'] # array to weights
  # for i in range(len(p)): Wnew[i] = p[i] # update the weights based on the candidate
  for i in range(len(p)): pdfnew.at[i,'weight'] = p[i] # update the weights based on the candidate
  # next read the
  fd,fnweight = tempfile.mkstemp(dir=mydir+'/batch')
  os.close(fd) # make sure closed
  #strc = 'nrniv -python ' + args['simf'] + ' '
  # save new synaptic weights to temp file
  pdfnew = pdfnew[pdfnew.time==np.amax(pdfnew.time)]  
  D = pdf2weightsdict(pdfnew)
  pickle.dump(D, open(fnweight,'wb')) # temp file for synaptic weights
  # next update the simulation's json file
  simconfig = args['simconfig']
  d = json.load(open(simconfig,'r')) # original input json
  simstr = d['sim']['name']
  print('args.keys():',args.keys())
  d['sim']['name'] += '_evo_gen_' + str(es.num_generations) + '_cand_' + str(cdx) +'_' # also need candidate ID
  d['sim']['doquit'] = 1
  d['sim']['doplot'] = 0
  d['simtype']['ResumeSim'] = 1
  d['simtype']['ResumeSimFromFile'] = fnweight
  fnjson = d['sim']['name'] + '.json'
  json.dump(d, open(fnjson,'w'), indent=2)
  print('fnjson is:',fnjson)
  ncore = 6
  strc = './myrun ' + str(ncore) + ' ' + fnjson # command string  
  return strc,'data/'+d['sim']['name']+'ActionsRewards.txt'

# evaluate fitness with sim run
def EvalFIT (candidates, args):
  global es
  print('EvalFIT args.keys():',args.keys())
  fitness = []; 
  for cdx, p in enumerate(candidates):
    strc,fn = FitJobStrFN(p, args, cdx)
    ret = os.system(strc)
    actreward = pd.DataFrame(np.loadtxt(fn),columns=['time','action','reward','proposed','hit','followtargetsign'])
    fitness.append(np.sum(actreward['reward']))
  return fitness

lconn,lssh=None,None

"""
# checks bounds for params
def my_bounder ():
  lmin,lmax=[],[]
  for k in dprm.keys():
    mn,mx = dprm[k].minval,dprm[k].maxval
    lmin.append(mn); lmax.append(mx)
  return ec.Bounder(lmin,lmax)
"""

weightVar = 0.75
wmin = 1.0 - weightVar
wmax = 1.0 + weightVar

# generate params 
@diversify
def my_generate (random, args):
  pout = []
  print('type:',type(args['startweight']))
  W = args['startweight']['weight']
  pdf = args['startweight']
  for i in range(len(W)):
    pout.append(random.uniform(wmin * pdf.at[i,'weight'], wmax * pdf.at[i,'weight']))
    #pout.append(random.uniform(wmin * W[i], wmax * W[i] ))
  return pout

# run sim command via mpi, then delete the temp file. returns job index and fitness.
def RunViaMPI (cdx, cmd, fn, maxfittime):
  global pc
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
  logger = logging.getLogger('inspyred.ec')
  logger.setLevel(logging.DEBUG)
  file_handler = logging.FileHandler('inspyred.log', mode='w')
  file_handler.setLevel(logging.DEBUG)
  formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
  file_handler.setFormatter(formatter)
  logger.addHandler(file_handler)
  return logger

# my_generate = sim.my_generate
# my_bounder = sim.my_bounder # when noBound==True, this is not guaranteed to be the same

# saves individuals in a population to binary file (pkl)
def my_indiv_observe (population, num_generations, num_evaluations, args):
  fn = 'data/' + evostr + '/gen_' + str(num_generations) + '_indiv.pkl'
  pickle.dump(population,open(fn,'w'))

es = None

"""
# adjusts boundary of parameters and saves new param bounds to file (pkl)
def my_bound_observe (population, num_generations, num_evaluations, args):
  if args['verbose']: print('adjusting param boundaries')
  sim.boundinc(population)
  #args['bounder'] = sim.my_bounder() # reset it here
  es.bounder = sim.my_bounder() # reset it here
  fn = 'data/' + evostr + '/dprm.pkl' # just saves to 1 to avoid extra files - dprm rarely changes
  pickle.dump(sim.dprm,open(fn,'w'))
"""

"""
@mutator
def my_mutation (random, candidate, args):
    # Return the mutants produced by nonuniform mutation on the candidates.
    #.. Arguments:
    #   random -- the random number generator object
    #   candidate -- the candidate solution
    #   args -- a dictionary of keyword arguments
    #Required keyword arguments in args:       
    #Optional keyword arguments in args:    
    #- *mutation_strength* -- the strength of the mutation, where higher
    #  values correspond to greater variation (default 1)
    #
    bounder = args['_ec'].bounder
    num_gens = args['_ec'].num_generations
    strength = args.setdefault('mutation_strength', 1)
    exponent = strength
    mutant = copy.copy(candidate)
    for i, (c, lo, hi) in enumerate(zip(candidate, bounder.lower_bound, bounder.upper_bound)):
        if random.random() <= 0.5:
            new_value = c + (hi - c) * (1.0 - random.random() ** exponent)
        else:
            new_value = c - (c - lo) * (1.0 - random.random() ** exponent)
        mutant[i] = new_value
    return mutant
"""
      
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
            nproc=16,rdmseed=1234,useDEA=True,\
            fstats='/dev/null',findiv='/dev/null',mutation_rate=0.2,useMPI=False,\
            numselected=100,noBound=False,simconfig='sn.json',\
            useLOG=False,maxfittime=600,lseed=None,larch=None,verbose=True,\
            useundefERR=False,startweight=None):
  global es
  if useLOG: logger=setEClog()
  rand = Random(); rand.seed(rdmseed) # alternative - provide int(time.time()) as seed
  if useDEA:
    es = ec.DEA(rand)
  else:
    es = ec.ES(rand)
  es.terminator = terminators.generation_termination 
  #es.variator = [inspyred.ec.variators.heuristic_crossover,my_mutation]#inspyred.ec.variators.nonuniform_mutation
  es.variator = [inspyred.ec.variators.heuristic_crossover,inspyred.ec.variators.nonuniform_mutation]
  es.observer = [my_indiv_observe] # saves individuals to pkl file each generation
  statfile = open(fstats,'w'); indfile = open(findiv,'w')
  es.observer.append(inspyred.ec.observers.file_observer)
  es.observer.append(inspyred.ec.observers.stats_observer)
  es.selector = inspyred.ec.selectors.tournament_selection
  es.replacer = inspyred.ec.replacers.generational_replacement#inspyred.ec.replacers.plus_replacement
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
                            max_generations=maxgen,
                            statistics_file=statfile,
                            individuals_file=indfile,
                            mutation_rate=mutation_rate,
                            simf=simf,
                            num_selected=numselected,
                            max_archive_size=max(popsize,numselected),
                            tournament_size=2,
                            num_elites=int(popsize/10.0),
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
  else: # use multiprocessing
    print('using multiprocessing')
    final_pop = es.evolve(generator=my_generate,
                            evaluator=inspyred.ec.evaluators.parallel_evaluation_mp,
                            mp_evaluator=EvalFIT, 
                            mp_nprocs=nproc,
                            pop_size=popsize,
                            maximize=True,
                            max_generations=maxgen,
                            statistics_file=statfile,
                            individuals_file=indfile,
                            mutation_rate=mutation_rate,
                            simf=simf,
                            num_selected=numselected,
                            max_archive_size=max(popsize,numselected),
                            tournament_size=2,
                            num_elites=int(popsize/10.0),
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
  # print(final_pop[0],final_pop[-1]))
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
  mutation_rate=0.2; evostr='21jul21A'; simconfig = 'sn.json'; maxfittime = 600; 
  noBound = useLOG = False;
  verbose = True; rdmseed=1234; useundefERR = False; 
  fseed = farch = lseed = larch = None; # files,lists for initial population and archive
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
        print('startweight columns:',startweight.columns)
    elif sys.argv[i] == '-python' or sys.argv[i] == '-mpi' or sys.argv[i] == 'evo.py':
      pass
    else: raise Exception('unknown arg:'+sys.argv[i])
    i+=1;

  if (useMPI and pc.id()==0) or not useMPI:
    print('popsize:',popsize,'maxgen:',maxgen,'nproc:',nproc,'useMPI:',useMPI,'numselected:',numselected,'evostr:',evostr,\
        'useDEA:',useDEA,'mutation_rate:',mutation_rate,\
        'noBound:',noBound,'useLOG:',useLOG,\
        'maxfittime:',maxfittime,'fseed:',fseed,'farch:',farch,'verbose:',verbose,'rdmseed:',rdmseed,'useundefERR:',useundefERR)
      
  # make sure master node does not work on submitted jobs (that would prevent it managing/submitting other jobs)
  if useMPI and pc.id()==0: pc.master_works_on_jobs(0) 

  if (useMPI and pc.id()==0) or not useMPI:
    # backup the config file and use backed-up version for evo (in case local version changed during evolution)
    safemkdir('data/'+evostr) # make a data output dir
    # simconfig = backupcfg(evostr,simconfig) 
    safemkdir(mydir+'/batch') # for temp files

  myout = runevo(popsize=popsize,maxgen=maxgen,nproc=nproc,rdmseed=rdmseed,useMPI=useMPI,\
                 numselected=numselected,mutation_rate=mutation_rate,\
                 useDEA=useDEA,fstats='/dev/null',findiv='/dev/null',simconfig=simconfig,\
                 useLOG=useLOG,maxfittime=maxfittime,lseed=lseed,larch=larch,\
                 verbose=verbose,useundefERR=useundefERR,startweight=startweight);

  if (useMPI and pc.id()==0) or not useMPI:
    pickle.dump(myout[0],open('data/' + evostr + '/fpop.pkl','w'))
    if useEMO: pickle.dump(myout[1],open('data/' + evostr + '/ARCH.pkl','w'))

  if useMPI:
    if pc.id()==0: print('MPI finished, exiting.')
    quit() # make sure CPUs freed
  
