import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np 
import h5py
import matplotlib.pyplot as plt
from functools import reduce, partial
import scipy.spatial
import os
import matplotlib
from matplotlib.colors import ListedColormap
import math
import multiprocessing

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
HUBBLE = 100 * 1.0e5 * PC * 1.0e-6 		# In seconds

#============================================================================================================================
# Mass and energy profiles 
#============================================================================================================================

def get_jet_length(z, jet_tracer, mass, jet_frac_min):
	jet_frac = jet_tracer / mass
	jet_frac[jet_frac<1.0e-30]=1.0e-30
	ids_jet = np.where(jet_frac > jet_frac_min)[0]
	z_jet = np.max(np.abs(z[ids_jet]))
	return z_jet

def get_single_mass_energy_profile(snapname, units, z_min, z_max_frac, z_max, num_bins, jet_frac_min):
						 
	UnitMass_in_g, UnitLength_in_cm, UnitVelocity_in_cm_per_s = units

	zjet_south, zjet_north = z_max

	print(f"Getting profiles for for {snapname}")
	
	with h5py.File(snapname, "r") as fh:
		# 
		pos_gas = np.asarray(fh["PartType0/Coordinates"], dtype = np.float64)
		vel_gas = np.asarray(fh["PartType0/Velocities"], dtype= np.float64)
		masses_gas = np.asarray(fh[f"PartType0/Masses"], dtype=np.float64)  
		uint_gas = np.asarray(fh[f"PartType0/InternalEnergy"], dtype=np.float64)
		disc_tracer_gas = np.asarray(fh[f"PartType0/DiscTracerField"], dtype=np.float64)
		jet_tracer_gas = np.asarray(fh[f"PartType0/JetTracerField"], dtype=np.float64)
		
		pos_bh = np.asarray(fh["PartType5/Coordinates"], dtype = np.float64)[0]
		
	# Derived fields
	cgm_tracer_gas = masses_gas - jet_tracer_gas - disc_tracer_gas

	egy_kin_gas = np.power(np.linalg.norm(vel_gas, axis=1), 2)

	jet_frac_gas = jet_tracer_gas / masses_gas
	jet_frac_gas[jet_frac_gas < 1.0e-30] = 1.0e-30

	# Re-centre on the black hole and convert to pc
	pos_gas -= pos_bh
	pos_gas *= UnitLength_in_cm * PC

	z_gas = pos_gas[:,2]

	# bin centres
	z_n_centre, dz_n = np.linspace (z_min, zjet_north, num_bins, retstep=True)
	z_s_centre, dz_s = np.linspace (zjet_south, -z_min, num_bins, retstep=True)

	# bin edges 
	z_n = np.append(z_n_centre - 0.5 * dz_n, z_n_centre[-1] + 0.5 * dz_n)
	z_s = np.append(z_s_centre - 0.5 * dz_n, z_s_centre[-1] + 0.5 * dz_n)

	# Restrict to cells in the jet
	ids_jet = np.where(jet_frac_gas > jet_frac_min)[0]

	jet_tracer_gas = jet_tracer_gas[ids_jet]
	disc_tracer_gas = disc_tracer_gas[ids_jet]
	cgm_tracer_gas = cgm_tracer_gas[ids_jet]

	egy_kin_gas = egy_kin_gas[ids_jet]
	uint_gas = uint_gas[ids_jet]
	z_gas = z_gas[ids_jet]

	# Mass in the northern lobe
	mass_jet_bin = np.zeros(len(z_n_centre), dtype=np.float64)
	mass_dsc_bin = np.zeros(len(z_n_centre), dtype=np.float64)
	mass_cgm_bin = np.zeros(len(z_n_centre), dtype=np.float64)

	# Determine the bins
	ids_zbin_n = np.digitize(z_gas, z_n)

	for i in range(1, len(z_n)):
		ids = np.where(ids_zbin_n == i)[0]
		mass_jet_bin[i-1] = np.sum(jet_tracer_gas[ids])
		mass_dsc_bin[i-1] = np.sum(disc_tracer_gas[ids])
		mass_cgm_bin[i-1] = np.sum(cgm_tracer_gas[ids])

	mass_tot_bin = mass_jet_bin + mass_dsc_bin + mass_cgm_bin

	mass_tot = np.sum(mass_tot_bin)

	mass_tot_bin /= mass_tot
	mass_jet_bin /= mass_tot
	mass_dsc_bin /= mass_tot
	mass_cgm_bin /= mass_tot

	# (Specific) energy in the southern lobe
	egy_tot_bin = np.zeros(len(z_s_centre), dtype=np.float64)
	egy_kin_bin = np.zeros(len(z_s_centre), dtype=np.float64)
	egy_thm_bin = np.zeros(len(z_s_centre), dtype=np.float64)

	# Determine the bins
	ids_zbin_s = np.digitize(z_gas, z_s)

	for i in range(1, len(z_s)):
		ids = np.where(ids_zbin_s == i)[0]
		egy_kin_bin[i-1] = np.sum(egy_kin_gas[ids])
		egy_thm_bin[i-1] = np.sum(uint_gas[ids])

	egy_tot_bin = egy_kin_bin + egy_thm_bin

	egy_tot = np.sum(egy_tot_bin)

	egy_tot_bin /= egy_tot
	egy_kin_bin /= egy_tot
	egy_thm_bin /= egy_tot

	return z_n_centre, mass_tot_bin, mass_jet_bin, mass_dsc_bin, mass_cgm_bin,\
		   z_s_centre, egy_tot_bin, egy_kin_bin, egy_thm_bin

#============================================================================================================================
# Slice
#============================================================================================================================

def wrap(x, boxsize, periodic):
	if periodic:
		x = x % boxsize
		if hasattr(x, "__len__"):
			x[x>0.5*boxsize] -= boxsize
			x[x< -0.5 * boxsize] += boxsize
		else:
			if x > 0.5 * boxsize:
				return x  - boxsize 
			else:
				if x< -0.5 * boxsize:
					return x + boxsize 
	return x

def create_temp_slice(snapname, extent_x, aspect, pixels, axes, units, time_jeton, jet_frac_min, periodic, z_max_frac):

	UnitMass_in_g, UnitLength_in_cm, UnitVelocity_in_cm_per_s = units

	xpix, ypix = pixels
	xmin, xmax = extent_x 

	print(f"Creating temp slice for {snapname}")
	
	with h5py.File(snapname, "r") as fh:
		pos_gas = np.asarray(fh["PartType0/Coordinates"], dtype= np.float64)
		masses_gas = np.asarray(fh["PartType0/Masses"], dtype=np.float64)
		
		uint_gas = np.asarray(fh["PartType0/InternalEnergy"], dtype=np.float64)
		temp_gas = GAMMA_MINUS1 * uint_gas * MU * MP / KB * (UnitVelocity_in_cm_per_s ** 2)
		meanweight = 4 / (8 - 5 * (1 - HYDROGEN_MASSFRAC))
		temp_gas[temp_gas > 1.0e4] *= (meanweight / MU)

		jet_tracer_gas = np.asarray(fh["PartType0/JetTracerField"], dtype=np.float64)
		jet_frac_gas = jet_tracer_gas / masses_gas
		jet_frac_gas[jet_frac_gas < 1.0e-30] = 1.0e-30

		del uint_gas

		pos_bh = np.asarray(fh["PartType5/Coordinates"], dtype= np.float64)[0]

		time = np.asarray(fh["Header"]. attrs.get("Time"), dtype = np.float32)
		boxsize = np.asarray(fh["Header"]. attrs.get("BoxSize"), dtype = np.float32)

	# Recentre the particles
	pos_gas -= pos_bh

	# Convert positions to parsecs
	pos_gas *= UnitLength_in_cm * PC
	boxsize *= UnitLength_in_cm * PC

	z_gas = pos_gas[:,2]

	z_jet = get_jet_length(z_gas, jet_tracer_gas, masses_gas, jet_frac_min)

	if z_max_frac * z_jet > xmax:
		xmax = z_max_frac * z_jet

	if -z_max_frac * z_jet < xmin:
		xmin = -z_max_frac * z_jet

	deltax =  0.5 * (xmax - xmin) 

	ymax = deltax / aspect
	ymin = -deltax / aspect

	extent = [xmin, xmax, ymin, ymax]

	time -= time_jeton
	time *= UnitLength_in_cm / UnitVelocity_in_cm_per_s * MYEAR
	
	# Width and height of image
	length_x = xmax - xmin
	length_y = ymax - ymin

	# Pixel sizes in sim coords
	dx = length_x / float(xpix)
	dy = length_y / float(ypix)

	# Search region
	xsmax = xmax + 0.1 * length_x
	xsmin = xmin - 0.1 * length_x
	ysmax = ymax + 0.1 * length_y
	ysmin = ymin - 0.1 * length_y
	zsmax = 0.01 * boxsize
	zsmin = - 0.01 * boxsize

	# Restrict to particles within the search region
	idsx = np.where((wrap(pos_gas[:,axes[0]], boxsize, periodic) >= xsmin) & (wrap(pos_gas[:,axes[0]], boxsize, periodic)  <= xsmax))[0]
	idsy = np.where((wrap(pos_gas[:,axes[1]], boxsize, periodic)  >= ysmin) & (wrap(pos_gas[:,axes[1]], boxsize, periodic)  <= ysmax))[0]
	idsz = np.where((wrap(pos_gas[:,axes[2]], boxsize, periodic) >= zsmin) & (wrap(pos_gas[:,axes[2]], boxsize, periodic)  <= zsmax))[0]

	mask = reduce(np.intersect1d, (idsx, idsy, idsz))

	pos_gas = pos_gas[mask]
	temp_gas = temp_gas[mask]

	# Initialise the image grid
	xgrid = np.linspace(xmin + 0.5 * dx, xmax - 0.5 * dx, xpix)
	ygrid = np.linspace(ymin + 0.5 * dy, ymax - 0.5 * dy, ypix)

	x,y,z = np.meshgrid(xgrid, ygrid, 0)

	gridpos = np.dstack((x.ravel(), y.ravel(), z.ravel()))

	pos = np.dstack((pos_gas[:,axes[0]],pos_gas[:,axes[1]],pos_gas[:,axes[2]]))[0]

	tree = scipy.spatial.cKDTree(pos)

	_, loc = tree.query(list(gridpos), k=1)

	temp_gas = temp_gas[loc].reshape(ypix,xpix)
	temp_gas = np.log10(temp_gas)

	return time, temp_gas,  extent


#============================================================================================================================
# Plotting
#============================================================================================================================

def setup_mpl():
	matplotlib.rc('font', family='sans-serif', size=8)
	matplotlib.rcParams['xtick.direction'] = 'in'
	matplotlib.rcParams['ytick.direction'] = 'in'
	matplotlib.rcParams['xtick.minor.visible'] = True
	matplotlib.rcParams['ytick.minor.visible'] = True
	matplotlib.rcParams['xtick.top'] = True
	matplotlib.rcParams['ytick.right'] = True
	matplotlib.rcParams['ytick.minor.left'] = True
	matplotlib.rcParams['ytick.minor.right'] = True
	matplotlib.rcParams['lines.dash_capstyle'] = "round"
	matplotlib.rcParams['lines.solid_capstyle'] = "round"
	matplotlib.rcParams['axes.linewidth'] = 0.3
	matplotlib.rcParams['xtick.labelsize']=6
	matplotlib.rcParams['ytick.labelsize']=6

	matplotlib.rcParams["xtick.major.size"] = 3
	matplotlib.rcParams["xtick.minor.size"] = 1.5
	matplotlib.rcParams["xtick.major.width"] = 0.6
	matplotlib.rcParams["xtick.minor.width"] = 0.4

	matplotlib.rcParams["ytick.major.size"] = 3
	matplotlib.rcParams["ytick.minor.size"] = 1.5
	matplotlib.rcParams["ytick.major.width"] = 0.6
	matplotlib.rcParams["ytick.minor.width"] = 0.4

	matplotlib.rcParams["axes.labelsize"] = 8


def get_sb_length(xmin, xmax):
	length_x = xmax - xmin
	sb_length = 0.1 * (xmax-xmin) 
	sb_length = int(math.ceil(sb_length/ 100.0)) * 100
	return 0.5 * sb_length

def plot_single_slice(snapnum, 
					  snappath, 
					  slice_kwargs,
					  me_kwargs,
					  plot_kwargs, 
					  savedir, 
					  units, 
					  time_jeton):
	
	snapname = f"{snappath}snap_{snapnum:03}.hdf5"
	
	time, temp, extent = create_temp_slice(snapname, **slice_kwargs)
	me_kwargs["z_max"] = (extent[0], extent[1])

	z_n, mass_tot, mass_jet, mass_dsc, mass_cgm,\
	z_s, egy_tot, egy_kin, egy_thm = get_single_mass_energy_profile(snapname, **me_kwargs)

	cmap = plot_kwargs["cmap"]
	vlims = plot_kwargs["vlims"]
	bbox = plot_kwargs["bbox"]
	figsize_x = plot_kwargs["figsize_x"]
	y_space = plot_kwargs["y_space"]
	cb_pad = plot_kwargs["cb_pad"]
	cb_frac_height = plot_kwargs["cb_frac_height"]
	width_cb = plot_kwargs["width_cb"]

	xmin, xmax, ymin, ymax = extent
	aspect = slice_kwargs["aspect"]
	z_min = me_kwargs["z_min"]

	setup_mpl()
	# Fraction of the figure taken up by the data axes (not inc the colorbars)
	datawidth_fig = bbox[1] - bbox[0]
	dataheight_fig = bbox[3] -bbox[2]

	dataheight_fig -= y_space

	width_slice = datawidth_fig
	width_mass = (xmax  - z_min) / (xmax - xmin) * width_slice
	width_egy = (-z_min - xmin) / (xmax - xmin) * width_slice
	width_space = 2. * z_min / (xmax - xmin) * width_slice

	height_slice = dataheight_fig * 5./8.
	height_mass = dataheight_fig * 3./8.
	height_egy = dataheight_fig * 3./8.
	height_cb = cb_frac_height * height_slice

	# Choose the figure height to maintain aspect ratio between the axes
	figsize_y = figsize_x * (width_slice / height_slice / aspect)

	# Where the data axes start (not the colorbars)
	dataaxes_start_x = bbox[0]
	dataaxes_start_y = bbox[2]

	print(f"Plotting for snap {snapnum}")

	fig = plt.figure(figsize=(figsize_x, figsize_y))

	ax_slice = fig.add_axes([dataaxes_start_x, dataaxes_start_y, width_slice, height_slice])
	ax_egy = fig.add_axes([dataaxes_start_x, dataaxes_start_y + height_slice + y_space, width_egy, height_egy])
	ax_mass = fig.add_axes([dataaxes_start_x + width_egy + width_space, dataaxes_start_y + height_slice + y_space, width_mass, height_mass])
	ax_cb = fig.add_axes([dataaxes_start_x + width_slice + cb_pad, dataaxes_start_y + 0.5 * (1 - cb_frac_height) * height_slice, width_cb, height_cb])

	ax_egy.plot(z_s, egy_tot, color="k", lw=0.7, ls="-")
	ax_egy.plot(z_s, egy_kin, color="#4E79A6", lw=0.7, ls="-")
	ax_egy.plot(z_s, egy_thm, color="#DF585B", lw=0.7, ls="-")

	ax_mass.plot(z_n, mass_tot, color="k", lw=0.7, ls="-")
	ax_mass.plot(z_n, mass_jet, color="#76B7B1", lw=0.7, ls="-")
	ax_mass.plot(z_n, mass_dsc, color="#F39BA8", lw=0.7, ls="-")
	ax_mass.plot(z_n, mass_cgm, color="#F3994C", lw=0.7, ls="-")
	
	norm = plt.Normalize(vmin = vlims[0], vmax=vlims[1])
	im = ax_slice.imshow(temp, cmap = cmap, extent = extent, norm = norm, origin = "lower")

	ax_slice.set_xlim(extent[0], extent[1])
	ax_slice.set_ylim(extent[2], extent[3])
	ax_slice.set_xticks([])
	ax_slice.set_yticks([])

	ax_slice.text(0.98,0.96, f"{time:.2f} Myr", color='w',fontsize=10,horizontalalignment='right',verticalalignment='top', transform=ax_slice.transAxes)
	
	sb_half_length = get_sb_length(xmin, xmax)
	sb_label = f"{int(2 * sb_half_length)} pc"
		
	ax_slice.plot([-sb_half_length, sb_half_length], [0.95 * ymin, 0.95 * ymin], lw = 2, c="w")
	text = ax_slice.text(0,0.95 * ymin + 0.02 * (ymax-ymin), sb_label, color="w", fontsize=10, horizontalalignment='center', verticalalignment ='bottom')

	ax_mass.set_xlim(left = z_min, right = xmax)
	ax_mass.set_ylim(bottom = 9.0e-6, top = 1)

	ax_egy.set_xlim(left = xmin, right = -z_min)
	ax_egy.set_ylim(bottom = 9.0e-6, top = 1)

	ax_mass.set_yscale("log")
	ax_egy.set_yscale("log")

	ax_mass.set_xticklabels([], minor=True)	
	ax_mass.set_xticklabels([], minor=False)
	ax_egy.set_xticklabels([], minor=True)	
	ax_egy.set_xticklabels([], minor=False)

	handles = [Line2D([0],[0], lw = 1.5, color="k", ls = "-", label = "total"),
			   Line2D([0],[0], lw = 1.5, color="#4E79A6", ls = "-", label = "kinetic"),
			   Line2D([0],[0], lw = 1.5, color="#DF585B", ls = "-", label = "thermal")]

	legend=ax_egy.legend(handles = handles, 
						loc = 'upper left', 
						bbox_to_anchor=(0.04,0.97), 
						fontsize = 4, 
						frameon=False, 
						labelspacing=0.7, 
						handlelength=2.5)

	handles = [Line2D([0],[0], lw = 1.5, color="k", ls = "-", label = "total"),
			   Line2D([0],[0], lw = 1.5, color="#F39BA8", ls = "-", label = "disc"),	
			   Line2D([0],[0], lw = 1.5, color="#76B7B1", ls = "-", label = "jet"),
			   Line2D([0],[0], lw = 1.5, color="#F3994C", ls = "-", label = "cgm")]

	legend=ax_mass.legend(handles = handles, 
						loc = 'upper right', 
						bbox_to_anchor=(0.96,0.97), 
						fontsize = 4, 
						frameon=False, 
						labelspacing=0.7, 
						handlelength=2.5)

	ax_egy.set_ylabel(r"$f_{\rm energy}$")
	ax_mass.set_ylabel(r"$f_{\rm mass}$")

	ax_mass.yaxis.set_label_position("right")
	ax_mass.yaxis.set_tick_params(labelright=True, labelleft=False)
	
	cbarlabel = r"${\rm log}_{10}(T \; [{\rm K}])}$"
	cb = matplotlib.colorbar.ColorbarBase(ax_cb, cmap = cmap, norm = norm, orientation="vertical")
	cb.set_label(cbarlabel, size=7, labelpad = 6)

	savename = f"{savedir}slice_{snapnum}.png"
	plt.savefig(savename, dpi=300)
	plt.close()

	print(f"Done for snap {snapnum}")


def create_plots(snappath, 
				 snapnums, 
				 slice_kwargs, 
				 me_kwargs, 
				 plot_kwargs,
				 units, 
				 time_jeton,
				 num_cores):


	# get a target function that only takes the group as an argument
	target = partial(plot_single_slice, snappath=snappath,
										slice_kwargs = slice_kwargs,
										me_kwargs = me_kwargs,
										plot_kwargs = plot_kwargs, 
										savedir = savedir, 
										units = units, 
										time_jeton = time_jeton)
	
	if not num_cores:
		num_cores = multiprocessing.cpu_count()

	workers = multiprocessing.Pool(processes = num_cores)

	np.array(workers.map(target, snapnums))

	workers.close()
	workers.join()
	print("Done :)")



#============================================================================================================================
# Units
#============================================================================================================================
UnitMass_in_g = 1.989e35					# 1.0e2 Msun
UnitVelocity_in_cm_per_s = 1.0e5			# 1 km/s
UnitLength_in_cm = 3.085678e18				# 1 pc

#============================================================================================================================
# Specific run parameters
#============================================================================================================================
snappath = "/put/your/path/to/snaps/here/"
snapnums = list(range(200,401))

vlims = [4.5,10.0]
savedir = "./jet_profile_movie/"
#============================================================================================================================
# General run parameters
#============================================================================================================================
time_jeton = 2.

z_min = 20.
z_max_frac = 1.1
z_max = 1000.

num_bins = 100
jet_frac_min = 1.0e-3

snapnums = list(range(200,401))
#============================================================================================================================
# Slice params
#============================================================================================================================
xpix = 1024
ypix = 1024

aspect = 3

axes_slice = [2,0,1]

periodic = True
#============================================================================================================================
# Plot params
#============================================================================================================================
figsize_x = 7
cmap = matplotlib.cm.inferno
bbox = [0.08,0.90,0.05,0.95]
y_space = 0.01

cb_pad = 0.008
cb_frac_height = 0.9
width_cb = 0.022
#============================================================================================================================
# Multiprocessing  
#============================================================================================================================
num_cores = 20
#============================================================================================================================
units = (UnitMass_in_g, UnitLength_in_cm, UnitVelocity_in_cm_per_s)
pixels = np.array([xpix,ypix], dtype=np.int)
extent_init = [-z_max, z_max]

slice_kwargs = {"extent_x": extent_init,
				"aspect": aspect,
			    "pixels": pixels,
			    "axes": axes_slice,
			    "units": units,
			    "time_jeton": time_jeton,
			    "jet_frac_min": jet_frac_min,
			    "periodic": periodic, 
			    "z_max_frac": z_max_frac }

me_kwargs = {"units": units,
			   "z_min": z_min, 
			   "z_max_frac": z_max_frac,
			   "num_bins": num_bins, 
			   "jet_frac_min": jet_frac_min}

plot_kwargs = {"figsize_x": figsize_x,
			   "cmap": cmap,
			   "vlims": vlims,
			   "bbox": bbox,
			   "y_space": y_space,
			   "cb_pad": cb_pad,
			   "cb_frac_height": cb_frac_height, 
			   "width_cb": width_cb}

if not os.path.isdir(savedir):
	os.makedirs(savedir)

create_plots(snappath, 
			 snapnums, 
			 slice_kwargs, 
			 me_kwargs, 
			 plot_kwargs, 
			 units, 
			 time_jeton, 
			 num_cores)




