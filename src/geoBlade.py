import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.interpolate import InterpolatedUnivariateSpline
from math import factorial
from src import geoAirfoil as ga
from joblib import Parallel, delayed

def BezierN(p, n):
    nc = 20
    t = np.linspace(0, 1, nc)
    B = 0
    for i in range(n+1):
        B += p[i]*t**i*(1-t)**(n-i)*(factorial(n)/(factorial(i)*factorial(n-i)))
    return B

def chordCurve(Pi, r, nameBlade):
    R = Pi[17]/2    
    pR_r = np.array([r[0], 0.5*(Pi[3]*R+r[0]), Pi[3]*R])
    pc_r = np.array([Pi[0]*Pi[17], Pi[1]*Pi[17], Pi[1]*Pi[17]])
    pR_t = np.array([Pi[3]*R, (0.485+0.5*Pi[3])*R, 0.97*R])
    pc_t = np.array([Pi[1]*Pi[17], Pi[1]*Pi[17], Pi[2]*Pi[17]])
    BR_r = BezierN(pR_r, 2)
    Bc_r = BezierN(pc_r, 2)
    BR_t = BezierN(pR_t, 2)
    Bc_t = BezierN(pc_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Bc = np.append(Bc_r[:-1], Bc_t)
    FC = InterpolatedUnivariateSpline(BR, Bc, k=4)
    c = FC(r)
    
    # font = {'family' : 'Liberation Serif',
    #         'weight' : 'normal',
    #         'size'   : 12}
    # cm=1/2.54
    # mpl.rc('font', **font)
    # mpl.rc('axes', linewidth=1)
    # mpl.rc('lines', lw=1)

    # fig = plt.figure(1, figsize=(12*cm, 4*cm))
    # ax = plt.subplot(111)
    # ax.plot(r/R, c/Pi[17], '-b')
    # # ax.plot(pR_r/R, pc_r/Pi[26], ':xr')
    # # ax.plot(pR_t/R, pc_t/Pi[26], ':xr')
    # for ns in range(len(r)):
    #     ax.plot(np.ones(2)*r[ns]/R, np.array([0, c[ns]])/Pi[17], '-g')
    # ax.set_aspect('equal')
    # ax.set_xlabel('$r/R$')
    # ax.set_ylabel('$c/D$')
    # ax.grid()
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_chord_curve.png')
    # plt.close(fig)
    # # plt.show()
    return c

def alphaCurve(Pi, r, R, nameBlade):
    pR_r = np.array([r[0], 0.5*(Pi[7]*R+r[0]), Pi[7]*R])
    palpha_r = np.array([Pi[4], Pi[5], Pi[5]])
    pR_t = np.array([Pi[7]*R, (0.485+0.5*Pi[7])*R, r[-1]])
    palpha_t = np.array([Pi[5], Pi[5], Pi[6]])
    BR_r = BezierN(pR_r, 2)
    Balpha_r = BezierN(palpha_r, 2)
    BR_t = BezierN(pR_t, 2)
    Balpha_t = BezierN(palpha_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Balpha = np.append(Balpha_r[:-1], Balpha_t)
    Falpha = InterpolatedUnivariateSpline(BR, Balpha, k=4)
    alpha = Falpha(r)
    
    # font = {'family' : 'Liberation Serif',
    #         'weight' : 'normal',
    #         'size'   : 12}
    # cm=1/2.54
    # mpl.rc('font', **font)
    # mpl.rc('axes', linewidth=1)
    # mpl.rc('lines', lw=1)

    # fig = plt.figure(2, figsize=(12*cm, 4*cm))
    # ax = plt.subplot(111)
    # ax.plot(r/R, alpha, '-b')
    # # ax.plot(pR_r/R, palpha_r, ':xr')
    # # ax.plot(pR_t/R, palpha_t, ':xr')
    # ax.set_xlabel('r/R')
    # ax.set_ylabel('$\\alpha$ [°]')
    # ax.grid()
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_alpha_curve.png')
    # plt.close(fig)   
    # # plt.show()
    return alpha

def airfoilCurves(Pi, r, R, nameBlade):    
    pR_r = np.array([r[0], 0.5*(Pi[14]*R+r[0]), Pi[14]*R])
    pds_r = np.array([Pi[8], Pi[10], Pi[10]])
    pR_t = np.array([Pi[14]*R, (0.485+0.5*Pi[14])*R, r[-1]])
    pds_t = np.array([Pi[10], Pi[10], Pi[12]])
    BR_r = BezierN(pR_r, 2)
    Bds_r = BezierN(pds_r, 2)
    BR_t = BezierN(pR_t, 2)
    Bds_t = BezierN(pds_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Bds = np.append(Bds_r[:-1], Bds_t)
    Fds = InterpolatedUnivariateSpline(BR, Bds, k=4)
    ds = Fds(r)
    
    # font = {'family' : 'Liberation Serif',
    #         'weight' : 'normal',
    #         'size'   : 12}
    # cm=1/2.54
    # mpl.rc('font', **font)
    # mpl.rc('axes', linewidth=1)
    # mpl.rc('lines', lw=1)
    
    # fig = plt.figure(3, figsize=(12*cm, 8*cm))
    # ax1 = plt.subplot(211)
    # ax1.plot(r/R, ds, '-b')
    # # ax1.plot(pR_r/R, pyt_r, ':xr')
    # # ax1.plot(pR_t/R, pyt_t, ':xr')
    # ax1.set_ylabel('$\Delta_{ds}$')
    # ax1.grid()

    pR_r = np.array([r[0], 0.5*(Pi[14]*R+r[0]), Pi[14]*R])
    pus_r = np.array([Pi[9], Pi[11], Pi[11]])
    pR_t = np.array([Pi[14]*R, (0.485+0.5*Pi[14])*R, r[-1]])
    pus_t = np.array([Pi[11], Pi[11], Pi[13]])
    BR_r = BezierN(pR_r, 2)
    Bus_r = BezierN(pus_r, 2)
    BR_t = BezierN(pR_t, 2)
    Bus_t = BezierN(pus_t, 2)
    BR = np.append(BR_r[:-1], BR_t)
    Bus = np.append(Bus_r[:-1], Bus_t)
    Fus = InterpolatedUnivariateSpline(BR, Bus, k=4)
    us = Fus(r)

    # ax2 = plt.subplot(212)
    # ax2.plot(r/R, us, '-b')
    # # ax2.plot(pR_r/R, pxt_r, ':xr')
    # # ax2.plot(pR_t/R, pxt_t, ':xr')
    # ax2.set_xlabel('r/R')
    # ax2.set_ylabel('$\Delta_{us}$')
    # ax2.grid()     
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_thickness_camber_Function.png')
    # plt.close(fig)
    # # plt.show()
    return ds, us

def paramBlade(Pi, nameBlade, r_pmin=0.2, r_pmax=0.97):
    r_R1 = np.linspace(r_pmin, 0.75, 22)
    r_R2 = np.linspace(0.776, r_pmax, 8)
    r_R = np.append(r_R1, r_R2)
    R = Pi[17]/2
    r = r_R*R
    c = chordCurve(Pi, r, nameBlade)
    alpha = alphaCurve(Pi, r, R, nameBlade)
    ds, us = airfoilCurves(Pi, r, R, nameBlade)
    
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)
    
    A = np.empty([0,12])
    xt = np.array([])
    yt = np.array([])
    xc = np.array([])
    yc = np.array([])
    # fig = plt.figure(7, figsize=(20*cm, 14*cm))
    for s in range(len(r)):
        As = ga.creatorAirfoil(ds[s], us[s])
        A = np.row_stack((A, As))
        X, YU, YL = ga.cstN5(As)
        # ax = plt.subplot(6, 5, s+1)
        # ax.plot(X, YU, '-k')
        # ax.plot(X, YL, '-k')
        # ax.set_aspect('equal')
        # ax.set_title('r/R = %s %%' % round(r_R[s]*100, 1))
        # ax.axis('off')
        X, YT, YC = ga.cst_ST_SC(As)
        xt = np.append(xt, X[np.argmax(YT)])
        yt = np.append(yt, max(YT))
        xc = np.append(xc, X[np.argmax(YC)])
        yc = np.append(yc, max(YC))
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_airfoils_blade.png')
    # plt.close(fig)
    # plt.show()
    
    np.save(nameBlade+'/A.npy', A)
    
    # fig = plt.figure(8, figsize=(12*cm, 16*cm))
    # ax1 = plt.subplot(411)
    # ax1.plot(r_R, xt, '-b')
    # ax1.set_ylabel('$x_t/c$')
    # ax1.grid()  
    # ax2 = plt.subplot(412)
    # ax2.plot(r_R, yt, '-b')
    # ax2.set_ylabel('$y_t/c$')
    # ax2.grid()
    # ax3 = plt.subplot(413)
    # ax3.plot(r_R, xc, '-b')
    # ax3.set_ylabel('$x_c/c$')
    # ax3.grid()
    # ax4 = plt.subplot(414)
    # ax4.plot(r_R, yc, '-b')
    # ax4.set_ylabel('$y_c/c$')
    # ax4.set_xlabel('$r/R$')
    # ax4.grid()
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_features_airfoils.png')
    # plt.close(fig)
    # plt.show()
    
    # dataSec = {'r'     : r,
    #            'r_R'   : r_R,
    #            'c'     : c,
    #            'alpha' : alpha,
    #            'ds'    : ds,
    #            'us'    : us,
    #            'A'     : A,
    #            'xt'    : xt,
    #            'yt'    : yt,
    #            'xc'    : xc,
    #            'yc'    : yc}
    
    dS = np.column_stack((A, r_R))
    dS = np.column_stack((dS, c))
    dS = np.column_stack((dS, alpha))
    dS = np.column_stack((dS, r))
    dS = np.column_stack((dS, ds))
    dS = np.column_stack((dS, us))
    dS = np.column_stack((dS, xt))
    dS = np.column_stack((dS, yt))
    dS = np.column_stack((dS, xc))
    dS = np.column_stack((dS, yc))
    
    return dS

def drawBlade(nameBlade, c, xt, r, phi, R, r_R):
    font = {'family' : 'Liberation Serif',
            'weight' : 'normal',
            'size'   : 10}
    cm=1/2.54
    mpl.rc('font', **font)
    mpl.rc('axes', linewidth=1)
    mpl.rc('lines', lw=1)

    le = -xt*c
    te = c-xt*c

    xle = le*np.cos(-phi*np.pi/180)
    yle = le*np.sin(-phi*np.pi/180)
    xte = te*np.cos(-phi*np.pi/180)
    yte = te*np.sin(-phi*np.pi/180)

    fig = plt.figure(16)
    ax = fig.add_subplot(projection='3d')
    ax.plot(np.zeros(2), np.zeros(2), np.array([0, R]), '--k')
    for ns in range(len(r)):
        ax.plot(np.array([xle[ns], xte[ns]]), np.array([yle[ns], yte[ns]]), np.ones(2)*r[ns], '-g')
    ax.plot(xle, yle, r, '-b')
    ax.plot(xte, yte, r, '-r')
    ax.set_box_aspect(aspect=(1, 0.5, 4), zoom=1)
    plt.tight_layout()
    fig.savefig(nameBlade+'/fig_blade.png')
    plt.close(fig)
    # plt.show()
    
    fig = plt.figure(17, figsize=(12*cm, 4*cm))
    ax = plt.subplot(111)
    ax.plot(r_R, phi, '-b')
    ax.set_xlabel('r/R')
    ax.set_ylabel('$\phi$ [°]')
    ax.grid()
    plt.tight_layout()
    fig.savefig(nameBlade+'/fig_phi_Function.png')
    plt.close(fig)
    # plt.show()