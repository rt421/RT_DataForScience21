import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import h5py
import os
from functools import reduce
from itertools import product

MSUN = 5.027e-34            # 1g in msol
PC = 3.2407e-19             # 1cm in pc
MYEAR = 3.17098e-14         # 1 Myr in seconds
YEAR = 3.17098e-8
GAMMA = 5./3.
GAMMA_MINUS1 = 5./3.-1.
HYDROGEN_MASSFRAC = 0.76
MU = 4.0 / (1. + 3. * HYDROGEN_MASSFRAC)
KB = 1.381e-16
MP = 1.6726e-24
GRAVITY = 6.674e-8
PI =  3.14159265359
CLIGHT = 2.99792458e10      # Speed of light
SIGMAT = 6.6524e-25         # Thomson cross-section
HUBBLE = 100 * 1.0e5 * PC * 1.0e-6      # In seconds


def get_histogram(x,y, weight, num_bins, lims):
    weight = weight / np.sum(weight)
    arr, _, _ = np.histogram2d(x, y,bins=num_bins,weights=weight, range=lims)
    arr[arr<=1.0e-30]=np.nan
    arr = arr.transpose()
    return arr

def get_hist(snappath, snap, units, rcut, fracs, num_bins, ranges):
                         
    UnitMass_in_g, UnitLength_in_cm, UnitVelocity_in_cm_per_s = units
    jf, df, cf = fracs
    
    snapname = f"{snappath}snap_{int(snap):03}.hdf5"
    
    with h5py.File(snapname, "r") as fh:
        pos_gas = np.asarray(fh["PartType0/Coordinates"], dtype= np.float64)
        vel_gas = np.asarray(fh["PartType0/Velocities"], dtype= np.float64)
        masses_gas = np.asarray(fh[f"PartType0/Masses"], dtype=np.float64)
        uint_gas = np.asarray(fh[f"PartType0/InternalEnergy"], dtype=np.float64)
        density_gas = np.asarray(fh[f"PartType0/Density"], dtype=np.float64) 
        jet_tracer_gas = np.asarray(fh[f"PartType0/JetTracerField"], dtype=np.float64)
        disc_tracer_gas = np.asarray(fh[f"PartType0/DiscTracerField"], dtype=np.float64)

        pos_bh = np.asarray(fh["PartType5/Coordinates"], dtype= np.float64)[0]
        vel_bh = np.asarray(fh["PartType5/Velocities"], dtype= np.float64)[0]
        
    # Derived fields
    pos_gas -= pos_bh
    pos_gas *= UnitLength_in_cm * PC

    r_gas = np.linalg.norm(pos_gas, axis=1)
    velz_gas = np.abs(vel_gas[:,2])

    temp_gas = GAMMA_MINUS1 * uint_gas * MU * MP / KB * (UnitVelocity_in_cm_per_s ** 2)
    meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))
    temp_gas[temp_gas > 1.0e4] *= (meanweight / MU)

    jet_frac_gas = jet_tracer_gas / masses_gas
    jet_frac_gas[jet_frac_gas < 1.0e-30] = 1.0e-30

    disc_frac_gas = disc_tracer_gas / masses_gas
    disc_frac_gas[disc_frac_gas < 1.0e-30] = 1.0e-30

    cgm_frac_gas = 1. - jet_frac_gas - disc_frac_gas
    cgm_frac_gas[cgm_frac_gas < 1.0e-30] = 1.0e-30

    # Delete fields that aren't needed
    del pos_gas, vel_gas, uint_gas, jet_tracer_gas, disc_tracer_gas

    # Masks
    ids = np.where(r_gas < rcut)[0]

    if jf != 0.:
        ids_jf = np.where(jet_frac_gas >= 10 ** jf)[0]
        ids = np.intersect1d(ids_jf, ids)
    if df != 0.:
        ids_df = np.where(disc_frac_gas >= df)[0]
        ids = np.intersect1d(ids_df, ids)
    if cf != 0.:
        ids_cf = np.where(cgm_frac_gas >= 10 ** cf)[0]
        ids = np.intersect1d(ids_cf, ids)

    r_gas = r_gas[ids]
    temp_gas = temp_gas[ids]
    density_gas = density_gas[ids]
    velz_gas = velz_gas[ids]
    masses_gas = masses_gas[ids]

    # Convert to physical units
    density_gas *= UnitMass_in_g / UnitLength_in_cm ** 3
    velz_gas *= UnitVelocity_in_cm_per_s * 1.0e-5

    # Take log of the fields
    temp_gas = np.log10(temp_gas)
    density_gas = np.log10(density_gas)
    velz_gas = np.log10(velz_gas)

    if jf == 0 and df == 0 and cf == 0:
        print('temp', np.min(temp_gas),np.max(temp_gas), np.mean(temp_gas), 'ranges', ranges["temp"])
        print('density', np.min(density_gas),np.max(density_gas), np.mean(density_gas), 'ranges', ranges["rho"])
        print('vel', np.min(velz_gas),np.max(velz_gas), np.mean(velz_gas), 'ranges', ranges["v"])

    # Create histograms
    trho_arr = get_histogram(temp_gas, density_gas, masses_gas, num_bins,[ranges["temp"], ranges["rho"]])
    tv_arr = get_histogram(temp_gas, velz_gas, masses_gas, num_bins,[ranges["temp"], ranges["v"]])
    rhov_arr = get_histogram(density_gas, velz_gas, masses_gas, num_bins,[ranges["rho"], ranges["v"]])

    return trho_arr, tv_arr, rhov_arr

#============================================================================================================================
# Units
#============================================================================================================================
UnitMass_in_g = 1.989e35                    # 1.0e2 Msun
UnitVelocity_in_cm_per_s = 1.0e5            # 1 km/s
UnitLength_in_cm = 3.085678e18              # 1 pc

units = (UnitMass_in_g, UnitLength_in_cm, UnitVelocity_in_cm_per_s)
#============================================================================================================================
# Run parameters
#============================================================================================================================
snappath = "/put/your/path/to/snaps/here/"
savedir = "./pdfs/"

snap = 300        # Snapshot number 
num_bins = 100
rcut = 3500       # [in pc] Do a spatial cut - don't want particles that are close to the edge of the box
#============================================================================================================================
# Tracer fracs
#============================================================================================================================
jet_fracs = [0.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
disc_fracs = [0.0, 0.75, 0.8, 0.85, 0.9, 0.95]
cgm_fracs = [0.0, -5.0, -4.5, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5]
#============================================================================================================================
# Ranges
#============================================================================================================================
t_range = [4.1,10.2]
rho_range = [-28.,-19.]
v_range = [-7.3, 4.7]
#============================================================================================================================
tracer_fracs = product(jet_fracs, disc_fracs, cgm_fracs)

ranges = {'temp': t_range, 
          "rho": rho_range, 
          "v": v_range}

ntot = len(jet_fracs) * len(disc_fracs) * len(cgm_fracs) - 1

if not os.path.exists(savedir):
    print(f"Creating {savedir}")
    os.makedirs(savedir)

savename = f"{savedir}/pdfs_snap_300.hdf5"

with h5py.File(savename, "w") as fh:
    fh.attrs.create("NumBins", num_bins)
    fh.attrs.create("TRange", t_range)
    fh.attrs.create("RhoRange", rho_range)
    fh.attrs.create("VRange", v_range)
    fh.attrs.create("RCut", rcut)

    for i, fracs in enumerate(tracer_fracs):
        #print(i, ntot)
    
        trho_arr, tv_arr, rhov_arr = get_hist(snappath, snap, units, rcut, fracs, num_bins, ranges)

        #print(f"({fracs[0]:.2f}, {fracs[1]:.2f}, {fracs[2]:.2f})") 

        grp = fh.create_group(f"({fracs[0]:.2f}, {fracs[1]:.2f}, {fracs[2]:.2f})")
        grp.create_dataset("Fracs", data = f"({fracs[0]:.2f}, {fracs[1]:.2f}, {fracs[2]:.2f})")
        grp.create_dataset("TRho", data = trho_arr, dtype = np.float64)    
        grp.create_dataset("TV", data = tv_arr, dtype = np.float64)
        grp.create_dataset("RhoV", data = rhov_arr, dtype = np.float64)

print("Done :)")            


