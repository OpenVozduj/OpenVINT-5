import numpy as np
from sklearn.preprocessing import MinMaxScaler
from nn import openvint_vae as av
from nn import openvint_mlp as am
from nn import readerGraphics as rg
import matplotlib.pyplot as plt
import matplotlib as mpl

def loadNN():
    vae = av.OV_VAE()
    vae.build(input_shape=(256, 256, 2))
    vae.load_weights('nn/vae_5.weights.h5')
    
    mlp = am.OV_MLP()
    mlp.load_weights('nn/mlp.weights.h5')
    return mlp, vae

def normParamCST():
    clarkY = np.load('src/clarky_cst5.npy')
    limited_down = clarkY - 0.4*abs(clarkY)
    limited_up = clarkY + 0.4*abs(clarkY)        
    XT = np.row_stack((clarkY, limited_down))
    XT = np.row_stack((XT, limited_up))
    scaler = MinMaxScaler(feature_range=(0, 1))
    normXT = scaler.fit_transform(XT)
    return normXT, scaler 

def coeffSections(scaler, mlp, vae, n_min, d, r_R, c, alpha, A,
                  nu, a):
    An = scaler.transform(A)
    zPred = mlp.predict(An)
    imgPred = vae.decoder.predict(zPred)
    n_s = n_min/60
    omega = 2*np.pi*n_s
    omegaR = omega*r_R*d/2
    Re = omegaR*c/nu
    M = omegaR/a
    
    cl = np.array([])
    cd = np.array([])
    for s in range(len(n_min)):
        cl_s, cd_s = rg.searchCoeffswithAlphaRe(alpha[s], Re[s], imgPred[s])
        cl = np.append(cl, cl_s)
        cd = np.append(cd, cd_s)
        
    # font = {'family' : 'Liberation Serif',
    #         'weight' : 'normal',
    #         'size'   : 10}
    # cm=1/2.54
    # mpl.rc('font', **font)
    # mpl.rc('axes', linewidth=1)
    # mpl.rc('lines', lw=1)
    # fig = plt.figure(11, figsize=(12*cm, 8*cm))
    # ax1 = plt.subplot(211)
    # ax1.plot(r_R, cl, '-b')
    # ax1.set_ylabel('$c_l$')
    # ax1.grid()
    # ax2 = plt.subplot(212)
    # ax2.plot(r_R, cd, '-b')
    # ax2.set_xlabel('r/R')
    # ax2.set_ylabel('$c_d$')
    # ax2.grid()
    # plt.tight_layout()
    # fig.savefig(nameBlade+'/fig_aeroCoeffs.png')
    # plt.close(fig)    
    # plt.show()
    ca = np.column_stack((cl, cd))
    ca = np.column_stack((ca, Re))
    ca = np.column_stack((ca, M))
    return ca

def thrustPower(n_min, nB, d, r, r_R, c, alpha,
                cl, cd, V_inf, rho, nu, a):
    R = d/2
    n_s = n_min/60
    lambda_p = V_inf/(d*n_s)
    omega = 2*np.pi*n_s
    omegaR = omega*r_R*d/2
    
    u1_R = np.zeros_like(r_R)
    integral_u1_R = np.zeros_like(r_R)    
    v_rel = V_inf/omegaR[-1]
    for l in range(10):
        v1_R = -v_rel/2 + np.sqrt((v_rel/2)**2 + u1_R*(r_R - u1_R) + 2*integral_u1_R)
        U1_R = r_R - u1_R
        V1_R = v_rel + v1_R
        W1_R = np.sqrt(U1_R**2 + V1_R**2)
        beta_1 = np.arctan(V1_R/U1_R)*(180/np.pi)
        phi = alpha + beta_1
        K = cl/cd
        invK = 1/K
        sigma = nB*c/(R*np.pi)
        Gamma_R = sigma*cl*W1_R/8
        f_R = (2/np.pi)*np.arccos(np.exp(-0.5*nB*(1-r_R)/(r_R*np.sin(beta_1*(np.pi/180)))))
        u1_R = Gamma_R/(f_R*r_R)
        u1_s = u1_R**2/r_R
        for ns in range(len(r_R)):
            integral_u1_R[ns] = np.trapz(u1_s[ns:], r_R[ns:])
    
    dCt = 8*Gamma_R*(U1_R - invK*V1_R)
    dmk = 8*Gamma_R*(V1_R + invK*U1_R)*r_R
    Ct = np.trapz(dCt, r_R)
    mk = np.trapz(dmk, r_R)
    Thrust =  0.5*Ct*rho*omegaR[-1]**2*np.pi*R**2
    Power = 0.5*mk*rho*omegaR[-1]**3*np.pi*R**2
    alpha_p = Thrust/(rho*n_s**2*d**4)
    beta_p = Power/(rho*n_s**3*d**5)
    eta_d = alpha_p*lambda_p/beta_p
    eta_s = Ct**(3/2)/(2*mk)
    H = 2*np.pi*r*np.tan(phi*np.pi/180)    
    
    TP = {}
    TP['T'] = Thrust
    TP['W'] = Power
    TP['eta_d'] = eta_d
    TP['eta_s'] = eta_s
    
    return TP, phi