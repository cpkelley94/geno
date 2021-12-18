from matplotlib import pyplot as plt
from scipy.integrate import odeint
import numpy as np
import matplotlib.cm as cm


#--  SYSTEMS OF ORDINARY DIFFERENTIAL EQUATIONS  ------------------------------#

def cas13d_dynamics_autoreg(v, t):
    '''
    This function represents the overall dynamics of Cas13d negative \\
    autoregulation by gRNA excision in cells, including transcription, \\
    translation, crRNA processing, and mRNA and protein degradation. This \\
    function is integrable by scipy.integrate.odeint().

    R = Cas13d-gRNA mRNA
    A = Cas13d apoprotein
    B = Cas13d:gRNA binary complex
    '''

    global r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B  # get rates

    # get concentrations in molecules/cell
    v_safe = [x if x > 0. else 0. for x in v]  # disallow negative concentrations
    c_R = v_safe[0]
    c_A = v_safe[1]
    c_B = v_safe[2]

    # define the diff eqs governing the state variables
    dR_dt = r_t - k_proc*c_R*c_A - k_deg_R*c_R   # transcription - crRNA processing - degradation
    dA_dt = k_T*c_R - k_proc*c_R*c_A - k_deg_A*c_A   # translation - crRNA processing - degradation
    dB_dt = k_proc*c_R*c_A - k_deg_B*c_B   # crRNA processing - degradation

    dv_dt = [dR_dt, dA_dt, dB_dt]

    # disallow moving into negative concentrations
    for ind, x in enumerate(v):
        if x <= 0 and dv_dt[ind] < 0:
            dv_dt[ind] = 0
    
    return dv_dt

def cas13d_dynamics_no_reg(v, t):
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

    global r_t, k_T, k_proc, k_deg_R, k_deg_A, k_deg_B, r_t_G, k_deg_G  # get rates

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


#--  SETUP  -------------------------------------------------------------------#

# constants
AVOGADRO = 6.022E23
VOL_CELL = 1.E-12  # [L]

# define default rates
r_t = 0.02  # [mRNA s^-1]
k_T = 0.083/10.  # [protein s^-1 mRNA^-1]
k_deg_R = np.log(2)/(10.*3600.)  # average mRNA half-life = 10 hr (human)
k_deg_A = np.log(2)/(36.*3600.)  # average protein half-life = 36 hr (HeLa)
k_deg_B = k_deg_A  # assumption: binary complex has same half-life and does not dissociate
k_proc = 1.E6/(AVOGADRO*VOL_CELL)  # [cell molec^-1 s^-1]
k_deg_G = np.log(2)/(2.*3600.)  # estimated gRNA half-life = 2 hr
r_t_G = 1.E4*k_deg_G  # average no. gRNA molecules in cell = 10000

# initialize simulation
timevec = np.linspace(0, 3600.*480, 5001)  # 480 hr simulation, 5000 timesteps
tol = 1.E-9


#--  SCREEN 1: GUIDE RNA PROCESSING RATE  -------------------------------------#

# test a range of gRNA processing rates
exp_range = np.linspace(-8, 0, num=65)
k_proc_range = k_proc*np.power(10.*np.ones_like(exp_range), exp_range)
data_k_proc = []

# iterate over gRNA processing rates
for k in k_proc_range:
    k_proc = k  # set this global temporarily

    # integrate GENO and unregulated ODEs
    sol_reg = odeint(cas13d_dynamics_autoreg, [0,0,0], timevec, rtol=tol, atol=tol)
    sol_noreg = odeint(cas13d_dynamics_no_reg, [0,0,0,0], timevec, rtol=tol, atol=tol)

    # record simulation outcome
    data_element = [k] + list(sol_reg[-1,:]) + list(sol_noreg[-1,:]) + [sol_reg[-1,2]/sol_noreg[-1,2]]
    data_k_proc.append(data_element)

data_k_proc = np.array(data_k_proc)

# plot gRNA processing screen and save as PDF
fig, ax = plt.subplots()
ax.plot(data_k_proc[:,0], data_k_proc[:,-1], 'g-', lw=0.8)
ax.set_xlabel(r'Rate of crRNA processing ($\mathrm{molec}^{-1} \; \mathrm{s}^{-1}$)')
ax.set_ylabel('Fraction of binary complex remaining\nwith negative autoregulation')
ax.set_xscale('log')
ax.set_yscale('log')
plt.tight_layout()
plt.savefig('screen_k_proc.pdf', dpi=300)
plt.close()

k_proc = 1.E6/(AVOGADRO*VOL_CELL)  # reset to original value


#--  SCREEN 2: TRANSLATION RATE, GUIDE RNA PROCESSING RATE  -------------------#

# initialize lists for holding simulation output
vals_k_T = []
vals_frac_remaining = []
vals_efficiency = []
vals_conc_binary_reg = []
vals_conc_binary_unreg = []

# iterate over a few gRNA processing rates (isolines)
for k_proc_exp in [1, 3, 5]:
    k_proc = (10.**k_proc_exp)/(AVOGADRO*VOL_CELL)  # calculate gRNA processing rate from exponent
    print(k_proc)

    # test a range of translation rates (x-axis)
    exp_range = np.linspace(-4, 1, num=129)
    k_T_range = k_T*np.power(10.*np.ones_like(exp_range), exp_range)
    data_k_T = []

    # iterate over translation rates
    for k in k_T_range:
        k_T = k  # set this global temporarily

        # integrate GENO and unregulated ODEs
        sol_reg = odeint(cas13d_dynamics_autoreg, [0,0,0], timevec, rtol=tol, atol=tol)
        sol_noreg = odeint(cas13d_dynamics_no_reg, [0,0,0,0], timevec, rtol=tol, atol=tol)

        # record simulation outcome
        data_element = [k] + list(sol_reg[-1,:]) + list(sol_noreg[-1,:]) + [sol_reg[-1,2]/sol_noreg[-1,2], (sol_noreg[-1,2]-sol_reg[-1,2])/sol_noreg[-1,2]]
        data_k_T.append(data_element)

    k_T = 0.083/10.  # reset to original value
    data_k_T = np.array(data_k_T)

    vals_k_T.append(data_k_T[:,0])
    vals_frac_remaining.append(data_k_T[:,-2])
    vals_conc_binary_reg.append(data_k_T[:,3])
    vals_conc_binary_unreg.append(data_k_T[:,6])
    vals_efficiency.append(data_k_T[:,-1])

# plot translation rate screen and save to PDF
fig, ax = plt.subplots(figsize=(2.75, 2.75))
colors = [cm.RdPu_r(i/float(len(vals_k_T))) for i in range(len(vals_k_T))]
for x, y, c in zip(vals_k_T, vals_frac_remaining, colors):
    ax.plot(x, y, '-', c=c, lw=0.8)
ax.set_xlabel(r'Cas13d translation rate ($\mathrm{protein} \; \mathrm{s}^{-1} \; \mathrm{mRNA}^{-1}$)')
ax.set_ylabel('Fraction of binary complex remaining\nwith negative autoregulation')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('screen_k_T_fracremaining.pdf', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(2.75, 2.75))
for x, y, c in zip(vals_k_T, vals_efficiency, colors):
    ax.plot(x, y, 'k-', c=c, lw=0.8)
ax.set_xlabel(r'Cas13d translation rate ($\mathrm{protein} \; \mathrm{s}^{-1} \; \mathrm{mRNA}^{-1}$)')
ax.set_ylabel('Autoregulation efficiency')
ax.set_xscale('log')
plt.savefig('screen_k_T_efficiency.pdf', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(2.75, 2.75))
for x, y, c in zip(vals_k_T, vals_conc_binary_reg, colors):
    ax.plot(x, y, 'k-', c=c, lw=0.8)
for x, y, c in zip(vals_k_T, vals_conc_binary_unreg, colors):
    ax.plot(x, y, 'k--', c=c, lw=0.8)
ax.set_xlabel(r'Cas13d translation rate ($\mathrm{protein} \; \mathrm{s}^{-1} \; \mathrm{mRNA}^{-1}$)')
ax.set_ylabel('[Cas13d:gRNA binary complex]')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('screen_k_T_concentration.pdf', dpi=300)
plt.close()

# reset parameter values
r_t_G = 1.E4*k_deg_G
k_proc = 1.E6/(AVOGADRO*VOL_CELL)  # [cell molec^-1 s^-1]
k_T = 0.083/10.  # [protein s^-1 mRNA^-1]


#--  SCREEN 3: TRANSCRIPTION RATE, GUIDE RNA TRANSCRIPTION RATE  --------------#

# initialize lists for holding simulation output
vals_r_t = []
vals_frac_remaining = []
vals_efficiency = []
vals_conc_binary_reg = []
vals_conc_binary_unreg = []

# iterate over a few gRNA transcription rates (isolines)
for rtG_exp in np.linspace(3, 4, num=3, endpoint=True):
    r_t_G = (10**rtG_exp)*k_deg_G  # calculate gRNA transcription rate from exponent
    print(r_t_G)

    # test a range of transcription rates (x-axis)
    exp_range = np.linspace(-4, 0, num=129)
    r_t_range = r_t*np.power(10.*np.ones_like(exp_range), exp_range)
    data_r_t = []

    # iterate over transcription rates
    for r in r_t_range:
        r_t = r  # set this global temporarily

        # integrate GENO and unregulated ODEs
        sol_reg = odeint(cas13d_dynamics_autoreg, [0,0,0], timevec, rtol=tol, atol=tol)
        sol_noreg = odeint(cas13d_dynamics_no_reg, [0,0,0,0], timevec, rtol=tol, atol=tol)

        # record simulation outcome
        data_element = [r] + list(sol_reg[-1,:]) + list(sol_noreg[-1,:]) + [sol_reg[-1,2]/sol_noreg[-1,2], (sol_noreg[-1,2]-sol_reg[-1,2])/sol_noreg[-1,2]]
        data_r_t.append(data_element)

    r_t = 0.02  # reset to original value
    data_r_t = np.array(data_r_t)

    vals_r_t.append(data_r_t[:,0])
    vals_frac_remaining.append(data_r_t[:,-2])
    vals_conc_binary_reg.append(data_r_t[:,3])
    vals_conc_binary_unreg.append(data_r_t[:,6])
    vals_efficiency.append(data_r_t[:,-1])

# plot transcription rate screen
fig, ax = plt.subplots(figsize=(2.75, 2.75))
colors = [cm.RdPu_r(i/float(len(vals_r_t))) for i in range(len(vals_r_t))]
for x, y, c in zip(vals_r_t, vals_efficiency, colors):
    ax.plot(x, y, '-', c=c, lw=0.8)
ax.set_xlabel(r'Cas13d transcription rate ($\mathrm{mRNA} \; \mathrm{s}^{-1}$)')
ax.set_ylabel('Autoregulation efficiency')
ax.set_xscale('log')
plt.savefig('screen_r_t_efficiency.pdf', dpi=300)
plt.close()

fig, ax = plt.subplots(figsize=(2.75, 2.75))
for x, y, c in zip(vals_r_t, vals_conc_binary_reg, colors):
    ax.plot(x, y, '-', c=c, lw=0.8)
for x, y, c in zip(vals_r_t, vals_conc_binary_unreg, colors):
    ax.plot(x, y, '--', c=c, lw=0.8)
ax.set_xlabel(r'Cas13d transcription rate ($\mathrm{mRNA} \; \mathrm{s}^{-1}$)')
ax.set_ylabel('[Cas13d:gRNA binary complex]')
ax.set_xscale('log')
ax.set_yscale('log')
plt.savefig('screen_r_t_concentration.pdf', dpi=300)
plt.close()