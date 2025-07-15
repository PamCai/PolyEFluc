import jax.numpy as jnp
import jax
from jax import grad, jit, vmap, lax, scipy, pmap, config
import numpy as np
import sys

def process_file(filename):
    f = open(filename,'r')
    params = []
    for x in f:
        params.append(float(x))
    return params

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python theory-rescaled-5comp.py <datafile>")
        sys.exit(1)
    datafile = sys.argv[1]
    suffix = datafile[:-4]
    print(suffix)
    save_folder = 'polyelectrolyte/PolyEFluc/polyelectrolyte-5-system/unitless-rescaled'
    params = process_file(save_folder + '/' + datafile)

start_from = int(params[13])
end_iter = int(params[14])

# Define parameters for polymer
N = int(params[0])
P_max = 100
b = params[2] # nm
L = b * N # nm
A = params[1] # nm^2
lc = params[3] # e/nm
vp = b * A
# Define parameters for ion
vi = params[4] # nm^3
r = np.cbrt(vi*3/4) # nm
ch = params[5] # e
# Define parameters for solvent
vs = params[6] # nm^3
chsep = params[7] # nm
chval = params[8] # e
# Input parameters
polymer_prop = np.zeros((2,5))
polymer_prop[0,:] = np.array([N,L,A,b,lc])
polymer_prop[1,:] = np.array([N,L,A,b,-lc])
ion_prop = np.zeros((2,3))
ion_prop[0,:] = np.array([vi,r,ch])
ion_prop[1,:] = np.array([vi,r,-ch])
solvent_prop = np.zeros((1,3))
solvent_prop[0,:] = np.array([vs,chsep,chval])
# Define Flory-Huggins parameters
chi_ps = params[9]/vs # nm^-3
chi = np.zeros((5,5))
chi[0,4] = chi_ps
chi[4,0] = chi[0,4]
chi[1,4] = chi_ps
chi[4,1] = chi[1,4]
# Define volume fractions
vfp = params[10]
vfi = params[11]
vfs = 1. - 2. * vfp - 2. * vfi
vf = np.array([vfp,vfp,vfi,vfi,vfs])

# Define constants
T = params[12] # K
kBT = 1.38064852E-23 * T * 1.e18 # kg nm^2 / s^2
e0 = 8.8541878128E-12   #F/m (or also called e0)
e0 = e0 * 3.8962564e10 # converts m to nm and C to e (e^2 s^2 kg^-1 nm^-3)
e = 1.602E-19 # conversion factor from e to C
e0er = e0 + (chsep*chval)**2*(vfs/vs)/(3*kBT) # e^2 s^2 kg^-1 nm^-3 (this is e0*er)
epsilon = e0er*kBT

# Define linearly spaced time corresponding to range I want
t = jnp.linspace(params[15],params[16],num=int(params[17])) #,dtype=jnp.float64)
# Define tb to be 1 so non-dimensionalized time is the same as dimensionalized time
tb = 1.
xi = 3. * jnp.pi ** 2 * kBT * tb / (N ** 2 * b ** 2) # kg/s
print('xi is ',xi)
t = t/tb # nondimensionalize t, if tb = 1 then this is the same as t
print('tb is ',tb)
xb = jnp.sqrt(N * b ** 2 / (3. * jnp.pi ** 2)) # nm
print('xb is ',xb, 'nm')

# define Gamma2
k = np.logspace(params[18],np.log10(params[19]),int(params[20])) # 1/nm

G_Tensor = np.load(save_folder + '/' + suffix + '_G2.npy') # 1/nm^3, 2x2 matrix
print('length of k vector is ',len(k))
# Save G2 vector and corresponding k vector
np.save(save_folder + '/' + suffix + '_G2_used',G_Tensor)
np.save(save_folder + '/' + suffix + '_k_vec_used',k)
print("Fluctuation Vectors Saved")
# Non-dimensionalize Gamma2 and k
k = xb * k
k2 = jnp.square(k)
K2 = jnp.dot(k,k)
G_Tensor = G_Tensor * xb ** 3
G2_CC = G_Tensor[:,0,0]
G2_CA = G_Tensor[:,0,1]
G2_CP = G_Tensor[:,0,2]
G2_CN = G_Tensor[:,0,3]
G2_AA = G_Tensor[:,1,1]
G2_AP = G_Tensor[:,1,2]
G2_AN = G_Tensor[:,1,3]
G2_PP = G_Tensor[:,2,2]
G2_PN = G_Tensor[:,2,3]
G2_NN = G_Tensor[:,3,3]
G2 = [G2_CC,G2_CA,G2_CP,G2_CN,G2_AA,G2_AP,G2_AN,G2_PP,G2_PN,G2_NN]
# Non-dimensionalize chi
chi = chi * xb ** 3

# fluctuation magnitude calculation
def fluc_mag_CC(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    p1 = 4. * chi_CS ** 2 / xb ** 6
    p2 = -8. * chi_CS * lc ** 2 / (epsilon * K2 * xb * A ** 2)
    p3 = 4. * lc ** 4 * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 4)
    return (p1 + p2 + p3)
def fluc_mag_CA(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CA = chi[0,1]
    chi_AS = chi[1,4]
    p1 = 4. * chi_CS * (chi_CS + chi_AS - chi_CA) / xb ** 6 
    p2 = -8. * chi_CS * lc * (-lc) / (epsilon * xb * K2 * A ** 2)
    p3 = 4. * lc ** 2 * (chi_CA - chi_CS - chi_AS) / (epsilon * xb * K2 * A ** 2)
    p4 = 8. * lc ** 3 * (-lc) * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 4)
    return (p1 + p2 + p3 + p4)
def fluc_mag_CP(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CP = chi[0,2]
    chi_PS = chi[2,4]
    p1 = 4. * chi_CS * (chi_CS - chi_CP + chi_PS) / xb ** 6
    p2 = -8. * chi_CS * lc * ch / (epsilon * K2 * xb * A * vi)
    p3 = 4. * lc ** 2 * (chi_CP - chi_CS - chi_PS) / (epsilon * K2 * xb * A ** 2)
    p4 = 8. * lc ** 3 * ch * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 3 * vi)
    return (p1 + p2 + p3 + p4)
def fluc_mag_CN(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CN = chi[0,3]
    chi_NS = chi[3,4]
    p1 = 4. * chi_CS * (chi_CS - chi_CN + chi_NS) / xb ** 6
    p2 = -8. * chi_CS * lc * (-ch) / (epsilon * xb * K2 * A * vi)
    p3 = 4. * lc ** 2 * (chi_CN - chi_CS - chi_NS) / (epsilon * xb * A ** 2 * K2)
    p4 = 8. * lc ** 3 * (-ch) * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 3 * vi)
    return (p1 + p2 + p3 + p4)
def fluc_mag_AA(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CA = chi[0,1]
    chi_AS = chi[1,4]
    p1 = (chi_CS + chi_AS - chi_CA) ** 2 / xb ** 6
    p2 = 4. * lc * (-lc) * (chi_CA - chi_CS - chi_AS) / (epsilon * xb * K2 * A ** 2)
    p3 = 4. * lc ** 2 * (-lc) ** 2 * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 4)
    return (p1 + p2 + p3)
def fluc_mag_AP(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CP = chi[0,2]
    chi_PS = chi[2,4]
    chi_CA = chi[0,1]
    chi_AS = chi[1,4]
    p1 = 2. * (chi_CS - chi_CP + chi_PS) * (chi_CS + chi_AS - chi_CA) / xb ** 6
    p2 = 4. * lc * (-lc) * (chi_CP - chi_CS - chi_PS) / (epsilon * xb * A ** 2 * K2) 
    p3 = 4. * lc * ch * (chi_CA - chi_AS - chi_CS) / (epsilon * xb * A * vi * K2) 
    p4 = 8. * lc ** 2 * (-lc) * ch * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 3 * vi)
    return (p1 + p2 + p3 + p4)
def fluc_mag_AN(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CN = chi[0,3]
    chi_NS = chi[3,4]
    chi_CA = chi[0,1]
    chi_AS = chi[1,4]
    p1 = 2. * (chi_CS - chi_CN + chi_NS) * (chi_CS + chi_AS - chi_CA) / xb ** 6
    p2 = 4. * lc * (-ch) * (chi_CA - chi_AS - chi_CS) / (epsilon * xb * A * vi * K2)
    p3 = 4. * lc * (-lc) * (chi_CN - chi_CS - chi_NS) / (epsilon * xb * A ** 2 * K2)
    p4 = 8. * lc ** 2 * (-lc) * (-ch) * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 3 * vi)
    return (p1 + p2 + p3 + p4)
def fluc_mag_PP(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CP = chi[0,2]
    chi_PS = chi[2,4]
    p1 = (chi_CS - chi_CP + chi_PS) ** 2 / xb ** 6
    p2 = 4. * lc * ch * (chi_CP - chi_CS - chi_PS) / (epsilon * xb * K2 * A * vi)
    p3 = 4. * lc ** 2 * ch ** 2 * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 2 * vi ** 2)
    return (p1 + p2 + p3)
def fluc_mag_PN(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CN = chi[0,3]
    chi_NS = chi[3,4]
    chi_CP = chi[0,2]
    chi_PS = chi[2,4]
    p1 = 2. * (chi_CS - chi_CN + chi_NS) * (chi_CS - chi_CP + chi_PS) / xb ** 6
    p2 = 4. * lc * (-ch) * (chi_CP - chi_CS - chi_PS) / (epsilon * xb * A * K2 * vi)
    p3 = 4. * lc * ch * (chi_CN - chi_CS - chi_NS) / (epsilon * xb * A * K2 * vi)
    p4 = 8. * lc ** 2 * ch * (-ch) * xb ** 4 / (epsilon ** 2 * K2 ** 2 * A ** 2 * vi ** 2)
    return (p1 + p2 + p3 + p4)
def fluc_mag_NN(chi, xb, lc, epsilon, K2, A, ch, vi):
    chi_CS = chi[0,4]
    chi_CN = chi[0,3]
    chi_NS = chi[3,4]
    p1 = (chi_CS - chi_CN + chi_NS) ** 2 / xb ** 6
    p2 = 4. * lc * (-ch) * (chi_CN - chi_CS - chi_NS) / (epsilon * K2 * xb * A * vi)
    p3 = 4. * lc ** 2 * (-ch) ** 2 * xb ** 4 / (epsilon ** 2 * A ** 2 * K2 ** 2 * vi ** 2)
    return (p1 + p2 + p3)

fm_CC = fluc_mag_CC(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_CA = fluc_mag_CA(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_CP = fluc_mag_CP(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_CN = fluc_mag_CN(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_AA = fluc_mag_AA(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_AP = fluc_mag_AP(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_AN = fluc_mag_AN(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_PP = fluc_mag_PP(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_PN = fluc_mag_PN(chi, xb, lc, epsilon, K2, A, ch, vi)
fm_NN = fluc_mag_NN(chi, xb, lc, epsilon, K2, A, ch, vi)
fm = [fm_CC,fm_CA,fm_CP,fm_CN,fm_AA,fm_AP,fm_AN,fm_PP,fm_PN,fm_NN]

FF = np.zeros((len(k),len(fm)))
for i in range(len(fm)):
    FF[:,i] = fm[i]*jnp.float_power(G2[i],-1.)
fluc_mag = jnp.sum(FF,axis=1)
np.save(save_folder + '/' + suffix + '_fluc_mag_5comp',fluc_mag)
integrated_overk = jnp.trapezoid(fluc_mag,x=k)
print('integration over fluc_mag is ',integrated_overk)

constant = 4. * jnp.pi * jnp.dot(k,k) * jnp.dot(k,k) * (A * b) ** 2 / (3. * (2. * jnp.pi) ** 3)
print('constant is ',constant)

# Define normal mode vector with smallest mode corresponding to 1 monomer
N_vec = jnp.arange(0,N,1)
P = jnp.arange(0,P_max,1)

# Define quantities needed for future calculations
phip = jnp.sqrt(2)*jnp.cos(jnp.pi/N*jnp.outer(N_vec,P)) # (nN, np)
phip = phip.at[:,0].set(1.0)
phip_squ = jnp.float_power(phip,2.)
NNP = jnp.einsum('ij,jk->ijk',phip,phip.T)
NNP = jnp.transpose(NNP,(1,0,2)) # (np, nN, nN)

######################################################
###### Iteration 0 - Rouse Correlation Function ######
######################################################
# Define function to compute Rouse correlation matrix for given t and p
@jit
def compute_CpRouse(pi, ti):
    #return (N / (jnp.pi ** 2 * P[pi] ** 2) * jnp.exp(-t[ti] * P[pi] ** 2))
    return 3. / (P[pi] ** 2) * jnp.exp(-t[ti] * P[pi] ** 2)

# Use vmap to vectorize the computation along t, p
CpRouse_vmap = jit(vmap(vmap(compute_CpRouse, in_axes=(0, None)), in_axes=(None,0)))

# Compute Rouse correlation function
if start_from == 0:
    CpR = CpRouse_vmap(jnp.arange(len(P)), jnp.arange(len(t)))
    CpR = CpR.at[:, 0].set(0.0) # (nt, np)
    jnp.save(save_folder+'/' + suffix + '_Cpiter0',CpR)

########################################################
### Iteration 0 - Structure Factor and Memory Kernel ###
########################################################
# Define function to compute Sptk
def compute_SpkRouse(pi, ki, ti):
    def compute_SpN(n):
        def compute_SpM(m):
            L1 = phip_squ[n, 1:] + phip_squ[m, 1:] # (np - 1)
            M1 = -1. / 6. * jnp.multiply(CpR[0, 1:], L1) # (np - 1)
            N1 = NNP[1:, n, m] # (np - 1)
            O1 = 1. / 3. * jnp.multiply(CpR[ti, 1:], N1) # (np - 1)
            Psum = jnp.sum(O1 + M1) # ()
            P1 = jnp.exp(k2[ki] * Psum - k2[ki] * t[ti])
            Q1 = NNP[pi, n, m] #phip[n, pi] * phip[m, pi]
            SpM = Q1 * P1
            return SpM

        SpM = vmap(compute_SpM)(jnp.arange(N))
        SpN = jnp.sum((SpM[:-1]+SpM[1:])*0.5)
        return SpN
    
    SpN = vmap(compute_SpN)(jnp.arange(N))
    Spk = jnp.sum((SpN[:-1]+SpN[1:])*0.5)
    return Spk

# Use vmap to vectorize the computation along p, k
SpkRouse_vmap = jit(vmap(vmap(compute_SpkRouse, in_axes=(0, None, None)), in_axes=(None, 0, None)))

if start_from == 0:
    Sptk = jnp.zeros((len(t),len(k),len(P)))
    # Compute Structure factor from Rouse correlation function by looping over t
    for ti in jnp.arange(len(t)):
        Sptk = Sptk.at[ti,:,:].set(SpkRouse_vmap(jnp.arange(len(P)), jnp.arange(len(k2)), ti)) # (nk, np)
    
    Sptk = jnp.transpose(Sptk,(0,2,1)) # (nt, np, nk)
    Sptk = Sptk / N ** 2
    
    print(np.max(Sptk))
    jnp.save(save_folder + '/' + suffix + '_Sptiter0',Sptk)

# Compute memory kernel from structure factor
if start_from == 0:
    Kptintegrand = jnp.einsum('ijk,k->ijk',Sptk,fluc_mag)
    Kpt = jnp.sum((Kptintegrand[:,:,:-1]+Kptintegrand[:,:,1:])*0.5*jnp.diff(k),axis=2) # (nt, np) # integrate over k
print('constant multiplying Kpt is ',constant)
if start_from == 0:
    Kpt = Kpt * constant
    print('Max of Kpt0 is ',jnp.max(Kpt))
    # Save memory kernel
    iter_num = 0
    jnp.save(save_folder+'/' + suffix + '_Kptiter{}'.format(iter_num),Kpt) 

######################################################
################ Find Gp(t) function #################
######################################################
# Define function to compute Gpt
def compute_Gp(pi, Kpt):
    # Set up grid for exp(p^2(t-t1)) for certain p
    exp_grid = jnp.exp(-1. * jnp.abs(t[:, jnp.newaxis] - t) * pi ** 2) # (nt,nt)

    def compute_Gt(ti):
        Gtintegrand = jnp.multiply(exp_grid[ti], Kpt[:,pi])
        Gt = jnp.cumsum((Gtintegrand[:-1]+Gtintegrand[1:])*0.5*(t[1]-t[0]))
        Gt = jnp.concatenate([jnp.array([0]),Gt])
        return Gt[ti]

    Gpt = vmap(compute_Gt)(jnp.arange(len(t)))
    return Gpt

ngpus = 4
pi_vec = jnp.arange(len(P))
pi_chunks = [pi_vec[i:i + ngpus] for i in range(0, len(P), ngpus)]

def compute_Gp_chunk(pi_chunk, Kpt):
    Gp_chunk = jnp.array([compute_Gp(pi, Kpt) for pi in pi_chunk])
    return Gp_chunk

Gp_pmap = pmap(compute_Gp_chunk, in_axes=(0, None))

# Compute Gpt
if start_from == 0:
    Gpt_chunks = Gp_pmap(pi_chunks, Kpt)
    print(jnp.shape(Gpt_chunks))
    Gpt_chunks_2 = jnp.transpose(Gpt_chunks,(1,0,2))
    Gpt = jnp.concatenate(Gpt_chunks_2,axis=0) # (np, nt)
    print(jnp.shape(Gpt))
    print('Max of Gp{} is '.format(iter_num),jnp.max(Gpt))
    # Save Gpt
    jnp.save(save_folder + '/' + suffix + '_Gptiter{}'.format(iter_num),Gpt)

######################################################
############### Find Gpinf function ##################
######################################################
# Define function to compute Gpt
@jit
def compute_Gpinf(pi, Kpt):
    # Set up grid for exp(p^2(t-t1)) for certain p
    exp_grid = jnp.exp(-1. * jnp.abs(t[:, jnp.newaxis] - t) * pi ** 2) # (nt,nt)
    Kpt_rev = jnp.flip(Kpt[:,pi])
    Gtintegrand = jnp.multiply(exp_grid[-1], Kpt_rev)
    Gt = jnp.cumsum((Gtintegrand[:-1]+Gtintegrand[1:])*0.5*(t[1]-t[0]))
    return Gt[-1]

# Use vmap to vectorize the computation along t, p
Gpinf_vmap = vmap(compute_Gpinf, in_axes=(0, None))

# Compute Gpinf
if start_from == 0:
    Gpinf = Gpinf_vmap(jnp.arange(len(P)), Kpt) # (np)
    
    # Save Gpinf
    jnp.save(save_folder + '/' + suffix + '_Gpinfiter{}'.format(iter_num),Gpinf)

######################################################
############## Find Hp(t) function ###################
######################################################
# Define function to calculate Kp matrix for certain p
def compute_Kpintegrand(t1, t2, pi, Kpt):
    index = jnp.absolute(t2 - t1)
    return Kpt[index.astype(int), pi]

# Define function to calculate H(t) at certain p
def compute_Ht(ti, Kpintegrand, exp_grid, t):
    Kptintegrand = jnp.einsum('ij,i->ij',Kpintegrand, exp_grid[ti])
    Kptintegrand = jnp.einsum('ij,j->ij',Kptintegrand, exp_grid[ti])
    Kptt1 = jnp.cumsum((Kptintegrand[:,:-1]+Kptintegrand[:,1:])*0.5*(t[1]-t[0]),axis=1)
    Kptt = jnp.diag(jnp.cumsum((Kptt1[:-1]+Kptt1[1:])*0.5*(t[1]-t[0]),axis=0))
    return Kptt[ti-1]

# Use vmap to vectorize computation along t1 and t2
Kpintegrand_vmap = jit(vmap(vmap(compute_Kpintegrand, in_axes=(0, None, None, None)), in_axes=(None, 0, None, None)))

# Use pmap to use separate gpus to speed up computation
ngpus = 4 # Size of each chunk (limited by ngpus)
ti_vec = jnp.arange(len(t))
ti_chunks = [ti_vec[i:i + ngpus] for i in range(0, len(t), ngpus)]

def compute_Ht_chunk(ti_chunk, Kpintegrand, exp_grid, t):
    # Compute Ht for the chunk
    Ht_chunk = jnp.array([compute_Ht(ti, Kpintegrand, exp_grid, t) for ti in ti_chunk])
    return Ht_chunk

# JIT compile the parallel version of the compute_Ht for chunks 
Ht_pmap = pmap(compute_Ht_chunk, in_axes=(0, None, None, None))

if start_from == 0:
    Hpt = jnp.zeros((len(P),len(t)))
    for pi in range(len(P)):
        # Compute Kp matrix
        Kpintegrand = Kpintegrand_vmap(jnp.arange(len(t)), jnp.arange(len(t)), pi, Kpt)
        
        # Set up grid for exp(p^2(t-t1)) for t1 = t1 and t2 and certain pi
        exp_grid = jnp.exp(-1. * jnp.abs(t[:, jnp.newaxis] - t) * pi ** 2)
    
        # Parallel compute Ht matrix for a certain p over chunks
        Ht_chunks = Ht_pmap(ti_chunks, Kpintegrand, exp_grid, t) # 4 chunks for 4 GPUs
        
        # Concatenate the results from the chunks
        Ht = jnp.concatenate(jnp.transpose(Ht_chunks),axis=0)
        # But need to replace t = 0 with 0 (t = 0 not actually evaluated in integral)
        Ht = Ht.at[0].set(0.0)
        
        # Store Ht
        Hpt = Hpt.at[pi,:].set(Ht)
    print('Max of Hp{} is '.format(iter_num),jnp.max(Hpt))
    jnp.save(save_folder + '/' + suffix + '_Hptiter{}'.format(iter_num),Hpt)

######################################################
######## Find Non-Rouse Correlation Function #########
######################################################
# Create inverted P^2 vector
P2inv = jnp.float_power(jnp.arange(len(P)),-2.)
P2inv = P2inv.at[0].set(0.0)

# Define function to compute non-Rouse Cp 
def compute_Cp(pi, ti, Kpt, Gpt, Gpinf, Hpt):
    return (3. / Kpt[0, pi] * Gpt.T[ti, pi] * Gpinf[pi] + 3. * (P2inv[pi] + Hpt.T[-1, pi]) * jnp.exp(-P[pi] ** 2 * t[ti]))

# Use vmap to vectorize the computation
Cp_vmap = jit(vmap(vmap(compute_Cp, in_axes=(0, None, None, None, None, None)), in_axes=(None, 0, None, None, None, None)))

# Compute correlation function
if start_from == 0:
    Cp = Cp_vmap(jnp.arange(len(P)), jnp.arange(len(t)), Kpt, Gpt, Gpinf, Hpt) # (nt, np)
    
    # Save correlation function
    iter_num = iter_num + 1
    jnp.save(save_folder + '/' + suffix + '_Cpiter{}'.format(iter_num),Cp)

######################################################
############### Creating K0(t) array #################
######################################################
# Define function to make square matrix of K0(t1-t2)
def compute_K0integrand(t1, t2, Kpt):
    index = jnp.absolute(t2 - t1)
    return Kpt[index.astype(int), 0]

# Use vmap to vectorize computation along t1 and t2
K0integrand_vmap = jit(vmap(vmap(compute_K0integrand, in_axes=(0, None, None)), in_axes=(None, 0, None)))

# Compute K0 matrix
if start_from == 0:
    K0integrand = K0integrand_vmap(jnp.arange(len(t)), jnp.arange(len(t)), Kpt) 
    
    K01 = jnp.cumsum((K0integrand[:,:-1]+K0integrand[:,1:])*0.5*(t[1]-t[0]),axis=1)
    K0 = jnp.diag(jnp.cumsum((K01[:-1]+K01[1:])*0.5*(t[1]-t[0]),axis=0))
    K0 = jnp.concatenate([jnp.array([0]), K0]) # (nt) need to add in t = 0 value
    
    # Save K0 array
    jnp.save(save_folder + '/' + suffix + '_K0iter{}'.format(iter_num),K0)

########################################################
####### Find Structure Factor and Memory Kernel ########
########################################################
# Define function to compute Sptk
def compute_Spk(pi, ki, ti, K0int, Cp):
    def compute_SpN(n):
        def compute_SpM(m):
            L1 = phip_squ[n, 1:] + phip_squ[m, 1:] 
            M1 = -1. / 6. * jnp.multiply(Cp[0, 1:], L1)
            N1 = NNP[1:, n, m] 
            O1 = 1. / 3. * jnp.multiply(Cp[ti, 1:], N1)
            Psum = jnp.sum(O1 + M1)
            P1 = jnp.exp(k2[ki] * Psum - k2[ki] * t[ti] - k2[ki] * 0.5 * K0int)
            Q1 = NNP[pi, n, m] # phip[n, pi] * phip[m, pi]
            SpM = Q1 * P1 
            return SpM

        SpM = vmap(compute_SpM)(jnp.arange(N))
        SpN = jnp.sum((SpM[:-1]+SpM[1:])*0.5)
        return SpN
    
    SpN = vmap(compute_SpN)(jnp.arange(N))
    Spk = jnp.sum((SpN[:-1]+SpN[1:])*0.5)
    return Spk

# Use vmap to vectorize the computation along k
Spk_vmap = vmap(compute_Spk, in_axes=(None, 0, None, None, None))

def compute_Spk_chunk(pi_chunk, ti, k2, K0int, Cp):
    Spk_chunk = jnp.array([Spk_vmap(pi, jnp.arange(len(k2)), ti, K0int, Cp) for pi in pi_chunk])
    return Spk_chunk

Spk_pmap = pmap(compute_Spk_chunk, in_axes=(0, None, None, None, None))

if start_from == 0:
    Sptk = jnp.zeros((len(t),len(P),len(k)))
    # Compute Structure factor from Rouse correlation function by looping over t
    for ti in jnp.arange(len(t)):
        K0int = K0[ti]
        #Sptk = Sptk.at[ti,:,:].set(SpkRouse_vmap(jnp.arange(len(P)), jnp.arange(len(k2)), ti)) # (nk, np)
        Spk_chunks = Spk_pmap(pi_chunks, ti, k2, K0int, Cp)
        Spk_chunks_2 = jnp.transpose(Spk_chunks,(1,0,2))
        Spk_full = jnp.concatenate(Spk_chunks_2,axis=0)
        Sptk = Sptk.at[ti,:,:].set(Spk_full)
    
    #Sptk = jnp.transpose(Sptk,(0,2,1)) # (nt, np, nk)
    Sptk = Sptk / N ** 2
    
    # Save Structure factor
    jnp.save(save_folder+'/' + suffix + '_Sptiter{}'.format(iter_num),Sptk)

    # Compute memory kernel from structure factor
    Kptintegrand = jnp.einsum('ijk,k->ijk',Sptk,fluc_mag)
    Kpt = jnp.sum((Kptintegrand[:,:,:-1]+Kptintegrand[:,:,1:])*0.5*jnp.diff(k),axis=2) # (nt, np)
    Kpt = Kpt*constant # nondimensionalize Kpt
    print('Max of Kpt{} is '.format(iter_num),jnp.max(Kpt))
    # Save memory kernel
    jnp.save(save_folder + '/' + suffix + '_Kptiter{}'.format(iter_num),Kpt)

if start_from != 0:
    iter_num = int(start_from - 1)
    Kpt = jnp.load(save_folder + '/' + suffix + '_Kptiter{}.npy'.format(iter_num))

while iter_num < end_iter:
    # Compute Gpt
    Gpt_chunks = Gp_pmap(pi_chunks, Kpt)
    Gpt_chunks_2 = jnp.transpose(Gpt_chunks,(1,0,2))
    Gpt = jnp.concatenate(Gpt_chunks_2,axis=0) # (np, nt)
    print('Max of Gp{} is '.format(iter_num),jnp.max(Gpt))
    # Save Gpt
    jnp.save(save_folder + '/' + suffix + '_Gptiter{}'.format(iter_num), Gpt)
    
    # Compute Gpinf
    Gpinf = Gpinf_vmap(jnp.arange(len(P)), Kpt) # (np)

    # Save Gpinf
    jnp.save(save_folder + '/' + suffix + '_Gpinfiter{}'.format(iter_num),Gpinf)
    
    Hpt = jnp.zeros((len(P),len(t)))
    for pi in range(len(P)):
        # Compute Kp matrix
        Kpintegrand = Kpintegrand_vmap(jnp.arange(len(t)), jnp.arange(len(t)), pi, Kpt)

        # Set up grid for exp(p^2(t-t1)) for t1 = t1 and t2 and certain pi
        exp_grid = jnp.exp(-1. * jnp.abs(t[:, jnp.newaxis] - t) * pi ** 2)

        # Parallel compute Ht matrix for a certain p over chunks
        Ht_chunks = Ht_pmap(ti_chunks, Kpintegrand, exp_grid, t) # 4 chunks for 4 GPUs
    
        # Concatenate the results from the chunks
        Ht = jnp.concatenate(jnp.transpose(Ht_chunks),axis=0)
        # But need to replace t = 0 with 0 (t = 0 not actually evaluated in integral)
        Ht = Ht.at[0].set(0.0)
    
        # Store Ht
        Hpt = Hpt.at[pi,:].set(Ht)
    print('Max of Hp{} is '.format(iter_num),jnp.max(Hpt))

    # Save Hp(t)
    jnp.save(save_folder + '/' + suffix + '_Hptiter{}'.format(iter_num),Hpt)
    
    # increase iter_num
    iter_num = iter_num + 1
    
    # Compute correlation function
    Cp = Cp_vmap(jnp.arange(len(P)), jnp.arange(len(t)), Kpt, Gpt, Gpinf, Hpt) # (nt, np)
    print('Max of Cp{} is '.format(iter_num),jnp.max(Cp))
    # Save correlation function
    jnp.save(save_folder + '/' + suffix + '_Cpiter{}'.format(iter_num),Cp)
    
    # Compute K0 matrix
    K0integrand = K0integrand_vmap(jnp.arange(len(t)), jnp.arange(len(t)), Kpt) 

    K01 = jnp.cumsum((K0integrand[:,:-1]+K0integrand[:,1:])*0.5*(t[1]-t[0]),axis=1)
    K0 = jnp.diag(jnp.cumsum((K01[:-1]+K01[1:])*0.5*(t[1]-t[0]),axis=0))
    K0 = jnp.concatenate([jnp.array([0]), K0]) # (nt)

    # Save K0 array
    jnp.save(save_folder + '/' + suffix + '_K0iter{}'.format(iter_num),K0)
    
    Sptk = jnp.zeros((len(t),len(P),len(k)))
    # Compute Structure factor from Rouse correlation function by looping over t
    for ti in jnp.arange(len(t)):
        K0int = K0[ti]
        Spk_chunks = Spk_pmap(pi_chunks, ti, k2, K0int, Cp)
        Spk_chunks_2 = jnp.transpose(Spk_chunks,(1,0,2))
        Spk_full = jnp.concatenate(Spk_chunks_2,axis=0)
        Sptk = Sptk.at[ti,:,:].set(Spk_full)
    Sptk = Sptk / N ** 2
    print('Max of Spt{} is '.format(iter_num),jnp.max(Sptk))
    # Save Structure factor
    jnp.save(save_folder+'/' + suffix + '_Sptiter{}'.format(iter_num),Sptk)

    # Compute memory kernel from structure factor
    Kptintegrand = jnp.einsum('ijk,k->ijk',Sptk,fluc_mag)
    Kpt = jnp.sum((Kptintegrand[:,:,:-1]+Kptintegrand[:,:,1:])*0.5*jnp.diff(k),axis=2) # (nt, np)
    Kpt = Kpt*constant
    print('Max of Kpt{} is '.format(iter_num),jnp.max(Kpt))
    # Save memory kernel
    jnp.save(save_folder + '/' + suffix + '_Kptiter{}'.format(iter_num),Kpt)
