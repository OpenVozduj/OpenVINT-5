#%% Libraries
import numpy as np
import os
from src import prePostFunc as pp
from src import operGenetics as og
from src import geoBlade as gb
from src import aeroPropeller as ap
import time

#%% Load neural network
mlp, vae = ap.loadNN()
_, scaler = ap.normParamCST()

#%% Preprocess
t0 = time.time()
fileInputs = 'inputs.json'
Omega, flow, opt = pp.readInputs(fileInputs)
rootDir = os.getcwd()
M = pp.initMetrics()

#%% Initial population
print('Start optimization \n')
print('Genration 0 \n')
os.mkdir(opt['testDir'])
fileGen = opt['testDir'] + '/G0'
os.mkdir(fileGen)

Pg, bladeg = og.initialPopulation(fileGen, opt['N'], Omega)
WTg = og.objectiveFunction(opt['N'], Pg, bladeg, scaler, mlp, vae, flow)

fn, v, d = og.distanceX(WTg[:,0], WTg[:,1], opt['Tmin'] )
p = og.twoPenalties(fn, v)
fp = p+d
M = og.updateMetrics(M, WTg[:,0], WTg[:,1], fp, WTg[:,2], WTg[:,3])
NEF = opt['N']

#%% Generations
for g in range(1, opt['G']):
    print('Genration '+str(g)+'\n')
    fileGen = opt['testDir'] + '/G' + str(g)
    os.mkdir(fileGen)
    Ng, NE = og.newSizePop(opt['N'], opt['G'], g)
    E = og.elitism(Pg, fp, NE)
    C = og.SBX(Pg, Ng-NE, fp)
    Q = og.mutation(C, Ng-NE, Omega)
    
    Pg = np.row_stack((E, Q))
    for i in range(Ng):
        bladeg[i] = fileGen+'/blade_'+str(i)
    WTg = og.objectiveFunction(Ng, Pg, bladeg, scaler, mlp, vae, flow)

    fn, v, d = og.distanceX(WTg[:,0], WTg[:,1], opt['Tmin'] )
    p = og.twoPenalties(fn, v)
    fp = p+d
    M = og.updateMetrics(M, WTg[:,0], WTg[:,1], fp, WTg[:,2], WTg[:,3])
    NEF += Ng
    if NEF >= 4500:
        break
    if M['error'][-1] <= opt['epsilon']:
        break
    
#%% Results
print('Optimization completed \n')
pp.printPlots(g, M, opt['testDir'])
pp.saveBladeOpt(fp, bladeg, opt['testDir'])
pp.blade3D(opt['testDir'])

timeCPU = time.time() - t0

