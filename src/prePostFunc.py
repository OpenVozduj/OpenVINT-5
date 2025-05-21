import numpy as np
import shutil
import json
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from src import geoAirfoil as ga

def readInputs(fileInputs):
    f = open(fileInputs)
    dataInput = json.load(f)    
    X = np.array([dataInput['intervals']['c_d_r'],
                  dataInput['intervals']['c_d_m'],
                  dataInput['intervals']['c_d_t'],
                  dataInput['intervals']['r_cdm'],
                  dataInput['intervals']['alpha_r'],
                  dataInput['intervals']['alpha_m'],
                  dataInput['intervals']['alpha_t'],
                  dataInput['intervals']['r_alpham'],
                  dataInput['intervals']['ds_r'],
                  dataInput['intervals']['us_r'],
                  dataInput['intervals']['ds_m'],
                  dataInput['intervals']['us_m'],
                  dataInput['intervals']['ds_t'],
                  dataInput['intervals']['us_t'],
                  dataInput['intervals']['r_am'],
                  dataInput['intervals']['n_min'],
                  dataInput['intervals']['B'],
                  dataInput['intervals']['d']])
    return X, dataInput['flow'], dataInput['optimization']

def initMetrics():
    metrics = {}
    metrics['error'] = np.array([])
    metrics['eta_d_opt'] = np.array([])
    metrics['eta_s_opt'] = np.array([])
    metrics['T_opt'] = np.array([])
    metrics['W_opt'] = np.array([])
    return metrics

def printPlots(g, metrics, testDir):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)

    fig = plt.figure(2, figsize=(16*cm, 16*cm))
    ax1 = fig.add_subplot(221)
    ax1.plot(np.arange(g+1), metrics['error'], '-k')
    ax1.set_xlabel('Generations')
    ax1.set_ylabel('Error [W]')
    ax1.set_yscale('log')
    ax1.grid()
    ax2 = fig.add_subplot(222)
    ax2.plot(np.arange(g+1), metrics['eta_d_opt'], '-g', label='$\eta_d$')
    ax2.plot(np.arange(g+1), metrics['eta_s_opt'], '-r', label='$\eta_s$')
    ax2.legend(loc='best')
    ax2.set_xlabel('Generations')
    ax2.set_ylabel('$\eta_{opt}$')
    ax2.grid()
    ax3 = fig.add_subplot(223)
    ax3.plot(np.arange(g+1), metrics['T_opt'], '-b')
    ax3.set_xlabel('Generations')
    ax3.set_ylabel('$T_{opt}$ [N]')
    ax3.grid()
    ax4 = fig.add_subplot(224)
    ax4.plot(np.arange(g+1), metrics['W_opt'], '-b')
    ax4.set_xlabel('Generations')
    ax4.set_ylabel('$W_{opt}$ [W]')
    ax4.grid()
    plt.tight_layout()
    fig.savefig(testDir + '/OpenVINT5_metrics.png')
    plt.show()
    
def saveBladeOpt(fp, bladeg, testDir):
    iopt = np.argmin(fp)
    shutil.copytree(bladeg[iopt], testDir+'/propellerOpt')
    
def blade3D(testDir):    
    os.mkdir(testDir+'/propellerOpt/airfoils_blade')
    
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)
    
    f = open(testDir+'/propellerOpt/info_prop.json')
    blade = json.load(f)
    
    A = np.load(testDir+'/propellerOpt/A.npy')
    
    fig = plt.figure(66)
    ax = fig.add_subplot(projection='3d')
    R = blade['design_parameters']['d']/2
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    
    for i in range(len(A)):
        X, YU, YL = ga.cstN5(A[i])
        XUI = np.delete(X, 0)
        YUI = np.delete(YU, 0)
        XUI = XUI[::-1]
        YUI = YUI[::-1]
        Xa = np.append(XUI, X, axis=0)
        Ya = np.append(YUI, YL, axis=0)
        Xscale = Xa*blade['info_sections']['c'][i] - blade['info_sections']['xt'][i]*blade['info_sections']['c'][i]
        Yscale = Ya*blade['info_sections']['c'][i]
        Xphi = Xscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180) - Yscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180)
        Yphi = Xscale*np.sin(-blade['info_sections']['phi'][i]*np.pi/180) + Yscale*np.cos(-blade['info_sections']['phi'][i]*np.pi/180)
        Z = np.ones_like(Xscale)*blade['info_sections']['r'][i]
        Points = np.column_stack((Xphi, Yphi))
        Points = np.column_stack((Points, Z))*1000
        ax.plot(Xphi, Yphi, Z, '-b')
        fileXFAirfoil = open(testDir+'/propellerOpt/airfoils_blade/airfoil_'+str(i)+'.dat','w')
        for i in range(Xphi.size):
            Pi = ''
            for k in range(3):
                Pi = Pi + str(np.round(Points[i,k], 6))+' '
            Pi = Pi +'\n'
            fileXFAirfoil.write(Pi)
        fileXFAirfoil.close()
    
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(testDir+'/propellerOpt/fig_blade.png')
    plt.close(fig)
    
