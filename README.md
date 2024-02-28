# aweSOM
Accelerated self-organizing map (SOM) and statistically combined ensemble (SCE)

This package combine a JIT-accelerated and parallelized implementation of SOM from [POPSOM](https://github.com/njali2001/popsom) and a GPU-accelerated implementation of SCE using [ensemble learning](https://github.com/mkruuse/segmenting-turbulent-simulations-with-ensemble-learning). It is optimized for plasma simulation data to segment current sheets in magneto-hydrodynamical (MHD) turbulence.

Barebone documentation is available inside individual scripts, and will be updated on this README soon. In the meantime, below is the basic pipeline to use aweSOM:

Start with a 3D simulation snapshot -----

1. To save features into a new numpy file, use "misc/obtain_features.py".
	
 Change the "outdir" in the configuration file (x16.ini) to the location of the snapshot
	
 In the command line, specify these arguments:
	
 --conf_path : path to configuration file
	
 --conf_name : name of configuration file
	
 --snapshot : the snapshot to obtain features from
	
 --scale : whether the dataset should be scaled using StandardScaler
	
 --crop : tuple, the start index of the cropped window and the width of the window
	
 -v/--var : list of features, format like this: "-v b_perp -v j_perp -v j_par"
	
  List of possible features:
    b_perp  bz  j_perp  j_par j_mag j_par_abs e_perp  e_par   e_dot_j  rho  b_perp_ciso  bz_ciso  j_perp_ciso  j_par_ciso  e_perp_ciso  e_par_ciso  e_dot_j_ciso   rho_ciso
  
  In the case where you call for 10+ features, having <100 GB of memory might not be enough, so a highmem node will be required (only take 10 minutes, so computational cost is minimal)
	

2. To compute the SOMs of the feature set, use "som/som3d.py". 
	
 In the command line, specify these arguments:
	
 --feature_path : path to the feature file
	
 --file : file name
	
 --xdim : x dimension of the map

 --ydim : y dimension of the map

 --alpha : learning rate

 --train :  number of training steps

 --batch : batch training, set as the size of the domain in pixel

 --pretrained : indicate whether to use a pretrained map

 --neurons_path : path to file containing the neuron values

 --save_neuron_values : beside outputting the clusters, include if you also want to save the map


3. In order to perform the SCE, you will need to run multiple instances of SOM with slightly different parameters. To generate this large dataset, use "parameter_scan.py". This does not run SOM multiple times, but instead generate a string that can be copied and pasted into a slurm/pbs submit script to run in parallel. Make sure the node has enough memory to support all instances (each loads the hdf5 file independently, which can be as big as 20 GB). Specify these arguments:
	
 --script_location : points to the location of the som3d

 --feature_path : path to the feature file

 --file : feature file name

 --save_neuron_values : beside outputting the clusters, True if you also want to save the map
	
	
 If you do not want batch training, specify "batch = [0]"
	
	
 An example of the submit script for the Rusty cluster is found in "batch_scripts/submit_highmem.cca"
	
 In rare cases, the code would crash for a specific set of parameters. This happens when the number of clusters is just 1


4. Create a folder called "SCE" within the current directory. Access SCE. You can start doing statistically combined ensemble now


5. To compute the 3D SCE, use "sce/sce3d.py" [not yet uploaded] or "sce/sce3d_jax.py", the latter of which was partially translated to JAX (https://jax.readthedocs.io/en/latest/index.html) for GPU- and JIT-accelerated computation. "sce3d_jax.py" also works with CPU-only system, but is often slower than non-JAX implementation due to additional overheads. In the case with 3D datacubes, it is best to use GPU for this task (the more memory the better; 100GB+ system and 80GB+ GPU on a 640^3 box). Only need to declare a few arguments when running:
	
 --folder : path to the encompassing folder
	
 --subfolder : name of the SCE subfolder
	
 An example of the submit script for the Rusty cluster is found in "batch_scripts/submit-sce3d.gpu

6. Once all SCE data has been obtained, it will return a number of npy files corresponding to the signal strength of each cluster in each file. It also returns a file called "multimap_mappings.txt" that saves the signal strength as a list of floats. Use this file as the input to "sce/parse_mapping3d.py", which sorts the list in descending order, take derivative to find breaks in the structure, and reassign combined cluster ids.

These are the arguments for "parse_mapping.py":

 --file_path : path to the SCE folder

 --copy_clusters : copy png files from the SCE folder to a subfolder named ranked-clusters, with the new names as its similarity rank

 --save_combined_map : after a threshold is found, use this argument to combine all files into one set of clustering result

 --threshold : separation threshold to find the trough of each valley in the derivate of gsum map

 --reference_file : if you want a plot output, provide the path to the feature file, which the code load to get the reference j_par data

 --slice : also for plotting; plot the slice at this location
 
 --ndim : the number of voxels in each dimension, for making the correct array to combine SCE clusters
 
 The first pass can be run with 1 core, since it's only reading in the data from one text file. Once a threshold has been identified, transfer the command over to a batch script to run on a node. The script still runs in serial, but the memory required is ~100 GB, so it's best to ask for a whole node if allocation is not a concern.
 
 Example of the submit script for this step is found in "batch_scripts/submit-map.cca"


  




