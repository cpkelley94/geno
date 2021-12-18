from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
import matplotlib.colors as colors

# constants
AVOGADRO = 6.022E23
VOL_CELL = 1.E-12  # [L]


#--  SYSTEMS OF ORDINARY DIFFERENTIAL EQUATIONS  ------------------------------#

def cas13d_dynamics_autoreg(v, t, r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B):
    '''
    This function represents the overall dynamics of Cas13d negative \\
    autoregulation by gRNA excision in cells, including transcription, \\
    translation, crRNA processing, and mRNA and protein degradation. This \\
    function is integrable by scipy.integrate.odeint().

    R = Cas13d-gRNA mRNA
    A = Cas13d apoprotein
    B = Cas13d:gRNA binary complex
    '''

    # get concentrations in molecules/cell
    v_safe = [x if x > 0. else 0. for x in v]  # disallow negative concentrations
    c_R = v_safe[0]
    c_A = v_safe[1]
    c_B = v_safe[2]

    # define the diff eqs governing the state variables
    dR_dt = r_t - k_proc*c_R*c_A - k_deg_R*c_R   # transcription - crRNA processing - degradation
    dA_dt = k_T*c_R - k_proc*c_R*c_A - k_deg_A*c_A   # translation - crRNA processing - degradation
    dB_dt = k_proc*c_R*c_A - k_deg_B*c_B  # crRNA processing - degradation

    dv_dt = [dR_dt, dA_dt, dB_dt]

    # disallow moving into negative concentrations
    for ind, x in enumerate(v):
        if x <= 0 and dv_dt[ind] < 0:
            dv_dt[ind] = 0

    return dv_dt

def cas13d_dynamics_no_reg(v, t, r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B, r_t_G, k_deg_G):
    '''
    This function represents the overall dynamics of Cas13d expression \\
    and complex formation without autoregulation, including transcription, \\
    translation, crRNA processing, and mRNA and protein degradation. This \\
    function is integrable by scipy.integrate.odeint().

    R = Cas13d mRNA
    A = Cas13d apoprotein
    B = Cas13d:gRNA binary complex
    G = unbound gRNA
    '''

    # get concentrations in molecules/cell
    v_safe = [x if x > 0. else 0. for x in v]  # disallow negative concentrations
    c_R = v_safe[0]
    c_A = v_safe[1]
    c_B = v_safe[2]
    c_G = v_safe[3]

    # define the diff eqs governing the state variables
    dR_dt = r_t - k_deg_R*c_R   # transcription - crRNA processing - degradation
    dA_dt = k_T*c_R - k_proc*c_G*c_A - k_deg_A*c_A   # translation - crRNA processing - degradation
    dB_dt = k_proc*c_G*c_A - k_deg_B*c_B   # crRNA processing - degradation
    dG_dt = r_t_G - k_proc*c_G*c_A - k_deg_G*c_G   # transcription - crRNA processing - degradation

    dv_dt = [dR_dt, dA_dt, dB_dt, dG_dt]

    # disallow moving into negative concentrations
    for ind, x in enumerate(v):
        if x <= 0 and dv_dt[ind] < 0:
            dv_dt[ind] = 0

    return dv_dt

def steady_state_Breg(r_t=0.02, k_T=0.0083, k_proc=1.E6/(AVOGADRO*VOL_CELL)):
    '''
    For a given set of parameters, integrate the GENO ODEs to calculate steady-\\
    state concentration of Cas13d binary complex.
    '''

    # define simulation parameters
    k_deg_R = np.log(2)/(10.*3600.)  # average mRNA half-life = 10 hr (human)
    k_deg_A = np.log(2)/(36.*3600.)  # average protein half-life = 36 hr (HeLa)
    k_deg_B = k_deg_A  # assumption: binary complex has same half-life and does not dissociate

    # simulation setup
    timevec = np.linspace(0, 3600.*480, 5001)
    tol = 1.E-9

    # simulate dynamics of regulated and unregulated systems
    sol_reg = odeint(cas13d_dynamics_autoreg, [0,0,0], timevec, args=(r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B), rtol=tol, atol=tol)
    B_reg = sol_reg[-1,2]

    return B_reg

def steady_state_efficiency(r_t=0.02, k_T=0.0083, k_proc=1.E6/(AVOGADRO*VOL_CELL)):
    '''
    For a given set of parameters, integrate the GENO ODEs to calculate the \\
    autoregulation effiency.
    '''

    # define simulation parameters
    k_deg_R = np.log(2)/(10.*3600.)  # average mRNA half-life = 10 hr (human)
    k_deg_A = np.log(2)/(36.*3600.)  # average protein half-life = 36 hr (HeLa)
    k_deg_B = k_deg_A  # assumption: binary complex has same half-life and does not dissociate
    k_deg_G = np.log(2)/(2.*3600.)
    r_t_G = 1.E4*k_deg_G

    # simulation setup
    timevec = np.linspace(0, 3600.*480, 5001)
    tol = 1.E-9

    # simulate dynamics of regulated and unregulated systems
    sol_reg = odeint(cas13d_dynamics_autoreg, [0,0,0], timevec, args=(r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B), rtol=tol, atol=tol)
    sol_noreg = odeint(cas13d_dynamics_no_reg, [0,0,0,0], timevec, args=(r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B, r_t_G, k_deg_G), rtol=tol, atol=tol)
    efficiency = (sol_noreg[-1,2]-sol_reg[-1,2])/sol_noreg[-1,2]

    return efficiency


#--  SETUP  -------------------------------------------------------------------#

# generate vectorized integration functions to run on meshgrids
ssBreg_vec = np.vectorize(steady_state_Breg)
sseff_vec = np.vectorize(steady_state_efficiency)

# set default parameter values
default_r_t = 0.02  # [mRNA s^-1]
default_k_T = 0.083/10.  # [protein s^-1 mRNA^-1]
default_k_proc = 1.E6/(AVOGADRO*VOL_CELL)  # [cell molec^-1 s^-1]

# set domains for heatmaps
rt_vals = np.logspace(-3, 0, endpoint=True, num=41)*default_r_t
kT_vals = np.logspace(-2, 1, endpoint=True, num=41)*default_k_T
kP_vals = np.logspace(-3, 0, endpoint=True, num=41)*default_k_proc


#--  GENERATE HEATMAPS  -------------------------------------------------------#

# heatmap: transcription vs translation, Breg
print('Breg: transcription vs. translation')
for kP, cond in [(kP_vals[0], 'low'), (kP_vals[-1], 'high')]:
    print('k_proc', cond, kP)
    rtkT, kTrt = np.meshgrid(rt_vals, kT_vals)
    Breg_rtkT = ssBreg_vec(r_t=rtkT, k_T=kTrt, k_proc=kP)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(rtkT, kTrt, Breg_rtkT, cmap='RdPu_r', norm=colors.LogNorm(vmin=1., vmax=1E4), rasterized=True, shading='gouraud')
    ax.axis([rtkT.min(), rtkT.max(), kTrt.min(), kTrt.max()])
    ax.set_xlabel('transcription rate')
    ax.set_ylabel('translation rate')
    ax.set_title(cond + ' kP')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/Breg_transcription_translation_' + cond + '_kP.pdf', dpi=300)
    plt.close()

# heatmap: translation vs crRNA processing, Breg
print('Breg: translation vs. processing')
for rt, cond in [(rt_vals[0], 'low'), (rt_vals[-1], 'high')]:
    print('r_t', cond, rt)
    kTkP, kPkT = np.meshgrid(kT_vals, kP_vals)
    Breg_kTkP = ssBreg_vec(k_T=kTkP, k_proc=kPkT, r_t=rt)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(kTkP, kPkT, Breg_kTkP, cmap='RdPu_r', norm=colors.LogNorm(vmin=1., vmax=1E4), rasterized=True, shading='gouraud')
    ax.axis([kTkP.min(), kTkP.max(), kPkT.min(), kPkT.max()])
    ax.set_xlabel('translation rate')
    ax.set_ylabel('crRNA proc rate')
    ax.set_title(cond + ' rt')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/Breg_translation_processing_' + cond + '_rt.pdf', dpi=300)
    plt.close()

# heatmap: crRNA processing vs transcription, Breg
print('Breg: processing vs. transcription')
for kT, cond in [(kT_vals[0], 'low'), (kT_vals[-1], 'high')]:
    print('k_T', cond, kT)
    kPrt, rtkP = np.meshgrid(kP_vals, rt_vals)
    Breg_kPrt = ssBreg_vec(k_proc=kPrt, r_t=rtkP, k_T=kT)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(kPrt, rtkP, Breg_kPrt, cmap='RdPu_r', norm=colors.LogNorm(vmin=1., vmax=1E4), rasterized=True, shading='gouraud')
    ax.axis([kPrt.min(), kPrt.max(), rtkP.min(), rtkP.max()])
    ax.set_xlabel('crRNA proc rate')
    ax.set_ylabel('transcription rate')
    ax.set_title(cond + ' kT')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/Breg_processing_transcription_' + cond + '_kT.pdf', dpi=300)
    plt.close()

# heatmap: transcription vs translation, efficiency
print('efficiency: transcription vs. translation')
for kP, cond in [(kP_vals[0], 'low'), (kP_vals[-1], 'high')]:
    print('k_proc', cond, kP)
    rtkT, kTrt = np.meshgrid(rt_vals, kT_vals)
    eff_rtkT = sseff_vec(r_t=rtkT, k_T=kTrt, k_proc=kP)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(rtkT, kTrt, eff_rtkT, cmap='GnBu_r', vmin=0.8, vmax=1, rasterized=True, shading='gouraud')
    ax.axis([rtkT.min(), rtkT.max(), kTrt.min(), kTrt.max()])
    ax.set_xlabel('transcription rate')
    ax.set_ylabel('translation rate')
    ax.set_title(cond + ' kP')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/efficiency_transcription_translation_' + cond + '_kP.pdf', dpi=300)
    plt.close()

# heatmap: translation vs crRNA processing, efficiency
print('efficiency: translation vs. processing')
for rt, cond in [(rt_vals[0], 'low'), (rt_vals[-1], 'high')]:
    print('r_t', cond, rt)
    kTkP, kPkT = np.meshgrid(kT_vals, kP_vals)
    eff_kTkP = sseff_vec(k_T=kTkP, k_proc=kPkT, r_t=rt)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(kTkP, kPkT, eff_kTkP, cmap='GnBu_r', vmin=0.8, vmax=1, rasterized=True, shading='gouraud')
    ax.axis([kTkP.min(), kTkP.max(), kPkT.min(), kPkT.max()])
    ax.set_xlabel('translation rate')
    ax.set_ylabel('crRNA proc rate')
    ax.set_title(cond + ' rt')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/efficiency_translation_processing_' + cond + '_rt.pdf', dpi=300)
    plt.close()

# heatmap: crRNA processing vs transcription, efficiency
print('efficiency: processing vs. transcription')
for kT, cond in [(kT_vals[0], 'low'), (kT_vals[-1], 'high')]:
    print('k_T', cond, kT)
    kPrt, rtkP = np.meshgrid(kP_vals, rt_vals)
    eff_kPrt = sseff_vec(k_proc=kPrt, r_t=rtkP, k_T=kT)

    fig, ax = plt.subplots(figsize=(3.5,3))
    c = ax.pcolormesh(kPrt, rtkP, eff_kPrt, cmap='GnBu_r', vmin=0.8, vmax=1, rasterized=True, shading='gouraud')
    ax.axis([kPrt.min(), kPrt.max(), rtkP.min(), rtkP.max()])
    ax.set_xlabel('crRNA proc rate')
    ax.set_ylabel('transcription rate')
    ax.set_title(cond + ' kT')
    ax.set_xscale('log')
    ax.set_yscale('log')
    fig.colorbar(c, ax=ax)
    ax.set_aspect('equal')
    plt.minorticks_off()
    plt.savefig('heatmaps/efficiency_processing_transcription_' + cond + '_kT.pdf', dpi=300)
    plt.close()

print(eff_rtkT.min())
print(eff_kTkP.min())
print(eff_kPrt.min())