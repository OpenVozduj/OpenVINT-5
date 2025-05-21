import numpy as np
from scipy.stats import qmc
from src import geoBlade as gb
from src import aeroPropeller as ap
import os
import json

def initialPopulation(fileGen, N, Omega):
    D = len(Omega)
    Dh = 0
    var = []
    for j in range(D):
        if Omega[j,0]!=Omega[j,1]:
            Dh += 1
            var.append(j)
    sampler = qmc.LatinHypercube(d=Dh)
    sample = sampler.random(n=N)
    LHx = qmc.scale(sample, Omega[var,0], Omega[var,1])
    Pg = np.zeros((N, D))
    idg = []
    for i in range(N):
        idg.append(fileGen+'/blade_'+str(i))
        k = 0
        for j in range(D):
            if (j in var) == True:
                Pg[i,j] = LHx[i,k]
                k += 1
            else:
                Pg[i,j] = Omega[j,0]
    Pg[:,16] = np.round(Pg[:,16])
    return Pg, idg


def objectiveFunction(N, Pg, nameBlade, scaler, mlp, vae, flow):
    n_min = np.array([])
    d = np.array([])
    nu = np.array([])
    a = np.array([])
    DS = np.empty([0,22])
    for i in range(N):
        os.mkdir(nameBlade[i])
        dS = gb.paramBlade(Pg[i], nameBlade[i])
        DS = np.row_stack((DS, dS))
        for s in range(30):
            n_min = np.append(n_min, Pg[i,15])
            d = np.append(d, Pg[i,17])
            nu = np.append(nu, flow['nu'])
            a = np.append(a, flow['a'])
    CA = ap.coeffSections(scaler, mlp, vae, n_min, d, DS[:,12],
                          DS[:,13], DS[:,14], DS[:,:12], nu, a)
    Wg = np.array([])
    Tg = np.array([])
    eta_dg = np.array([])
    eta_sg = np.array([])
    for i in range(N):
        ca = CA[30*i:30*(i+1)]
        ds = DS[30*i:30*(i+1)]
        TP, phi = ap.thrustPower(Pg[i,15], Pg[i,16], Pg[i,17], ds[:,15], 
                                 ds[:,12], ds[:,13], ds[:,14], ca[:,0], 
                                 ca[:,1], **flow)
        
        Wg = np.append(Wg,TP['W'])
        Tg = np.append(Tg,TP['T'])
        eta_dg = np.append(eta_dg, TP['eta_d'])
        eta_sg = np.append(eta_sg, TP['eta_s'])
    
        info_prop = {}
        info_prop['nameCase'] = nameBlade[i]
        info_prop['Thrust'] = TP['T']
        info_prop['Power'] = TP['W']
        info_prop['eta_d'] = TP['eta_d']
        info_prop['eta_s'] = TP['eta_s']
        info_prop['V_inf'] = flow['V_inf']
        design_param = {}
        design_param['c_d_r'] = Pg[i,0]
        design_param['c_d_m'] = Pg[i,1]
        design_param['c_d_t'] = Pg[i,2]
        design_param['r_cdm'] = Pg[i,3]
        design_param['alpha_r'] = Pg[i,4]
        design_param['alpha_m'] = Pg[i,5]
        design_param['alpha_t'] = Pg[i,6]
        design_param['r_alpham'] = Pg[i,7]
        design_param['ds_r'] = Pg[i,8]
        design_param['us_r'] = Pg[i,9]
        design_param['ds_m'] = Pg[i,10]
        design_param['us_m'] = Pg[i,11]
        design_param['ds_t'] = Pg[i,12]
        design_param['us_t'] = Pg[i,13]
        design_param['r_am'] = Pg[i,14]
        design_param['n_min'] = Pg[i,15]
        design_param['B'] = Pg[i,16]
        design_param['d'] = Pg[i,17]
        info_prop['design_parameters'] = design_param
        sections_data = {}
        sections_data['r_R'] = ds[:,12].tolist()
        sections_data['r'] = ds[:,15].tolist()
        sections_data['c'] = ds[:,13].tolist()
        sections_data['phi'] = phi.tolist()
        sections_data['xt'] = ds[:,18].tolist()
        sections_data['yt'] = ds[:,19].tolist()
        sections_data['xc'] = ds[:,20].tolist()
        sections_data['yc'] = ds[:,21].tolist()
        sections_data['Re'] = ca[:,2].tolist()
        sections_data['M'] = ca[:,3].tolist()
        sections_data['cla'] = ca[:,0].tolist()
        sections_data['cda'] = ca[:,1].tolist()
        info_prop['info_sections'] = sections_data
        section075 = {}
        section075['r_R'] = ds[21,12]
        section075['r'] = ds[21,15]
        section075['c'] = ds[21,13]
        section075['phi'] = phi[21]
        section075['xt'] = ds[21,18]
        section075['yt'] = ds[21,19]
        section075['xc'] = ds[21,20]
        section075['yc'] = ds[21,21]
        section075['Re'] = ca[21,2]
        section075['M'] = ca[21,3]
        section075['cla'] = ca[21,0]
        section075['cda'] = ca[21,1]
        info_prop['section0_75'] = section075
        
        json_object = json.dumps(info_prop, indent=11)      
        with open(nameBlade[i] + '/info_prop.json', 'w') as outfile:
            outfile.write(json_object)
            
    WTg = np.column_stack((Wg, Tg, eta_dg, eta_sg))
    
    return WTg

def distanceX(Wg, Tg, Tmin):
    # Normalize functions
    fn = np.array([])
    for i in range(len(Wg)):
        fn = np.append(fn, (Wg[i]-min(Wg))/(max(Wg)-min(Wg)))
    # Constrain functions
    g = Tmin - Tg
    # Constrain violations
    c = np.array([])
    for i in range(len(fn)):
        c = np.append(c, max(0, g[i]))
    v = np.array([])
    for i in range(len(fn)):
        v = np.append(v, c[i]/max(c))
    # distance
    rf = len(np.where(v==0)[0])/len(fn)
    d = np.zeros_like(fn)
    for i in range(len(fn)):
        if rf == 0:
            d[i] = v[i]
        else:
            d[i] = np.sqrt(fn[i]**2+v[i]**2)
    return fn, v, d

def twoPenalties(fn, v):
    rf = len(np.where(v==0)[0])/len(fn)
    X = np.array([])
    for i in range(len(fn)):
        if rf==0:
            X = np.append(X, 0)
        else:
            X = np.append(X, v[i])
    Y = np.zeros_like(fn)
    for i in range(len(fn)):
        if v[i] == 0:
            Y[i] = 0
        else:
            Y[i] = fn[i]
    p = np.zeros_like(fn)
    for i in range(len(fn)):
        p[i] = (1-rf)*X[i]+rf*Y[i]
    return p

def newSizePop(N0, G, g, alpha=5, Nend=20):
    c = -np.log((Nend-alpha)/(N0+alpha))/G
    Ng = round((N0+alpha)*np.exp(-c*g)-alpha)
    if Ng < Nend:
        Ng = Nend
    if Ng%2 == 0:
        NE = 2
    else:
        NE = 3
    return Ng, NE

def tournament(N, fp, pt=0.8):
    xr = np.random.choice(np.arange(N), 2, replace=False)
    r = np.random.rand()
    if r < pt:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[0]
        else:
            p = xr[1]
    else:
        if fp[xr[0]] < fp[xr[1]]:
            p = xr[1]
        else:
            p = xr[0]
    return p

def SBX(Pg, N, fp, eta_c=20):
    D = 18
    C = np.empty([0, D])
    for i in range(int(N/2)):
        while 1==1:
            p1 = tournament(N, fp)
            p2 = tournament(N, fp)
            if p1!=p2:
                break
        P1 = Pg[p1]
        P2 = Pg[p2]
        C1 = np.zeros_like(P1)
        C2 = np.zeros_like(P2)
        for j in range(D):
            u = np.random.uniform()
            if u < 0.5:
                beta = (2*u)**(1/(eta_c+1))
            else:
                beta = 1/((2*(1-u))**(1/(eta_c+1)))
            C1[j] = 0.5*((1-beta)*P1[j] + (1+beta)*P2[j])
            C2[j] = 0.5*((1+beta)*P1[j] + (1-beta)*P2[j])
        C = np.row_stack((C, C1, C2))
    return C

def mutation(C, N, Omega, eta_m=20):
    D = 18
    Q = np.empty([0, D])
    for i in range(N):
        p = C[i]
        m = np.zeros_like(p)
        for j in range(D):
            r = np.random.uniform()
            if r < 0.5:
                delta = (2*r)**(1/(eta_m+1))-1
            else:
                delta = 1 - (2*(1-r))**(1/(eta_m+1))
            m[j] = p[j] + delta*(Omega[j,1] - Omega[j,0])
        m = np.clip(m, Omega[:,0], Omega[:,1])
        m[16] = round(m[16])
        Q = np.row_stack((Q, m))
    return Q

def elitism(Pg, fp, NE):
    P = Pg.copy()
    f = fp.copy()
    E = np.empty([0, 18])
    for i in range(NE):
        iB = np.argmin(f)
        E = np.row_stack((E, P[iB]))
        P = np.delete(P, iB, axis=0)
        f = np.delete(f, iB)        
    return E

def updateMetrics(metrics, Wg, Tg, fp, eta_dg, eta_sg):
    iopt = np.argmin(fp)
    metrics['error'] = np.append(metrics['error'], abs(np.mean(Wg)-Wg[iopt]))
    metrics['eta_d_opt'] = np.append(metrics['eta_d_opt'], eta_dg[iopt])
    metrics['eta_s_opt'] = np.append(metrics['eta_s_opt'], eta_sg[iopt])
    metrics['T_opt'] = np.append(metrics['T_opt'], Tg[iopt])
    metrics['W_opt'] = np.append(metrics['W_opt'], Wg[iopt])
    return metrics