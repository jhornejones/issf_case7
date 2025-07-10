import subprocess
from pathlib import Path
import json

DEFAULT_PARAM_JSON = "params.json"
PARAM_OUTPUT_FILE = "results.json"

#-------------------------------------------------------------------------
#------------------------- USER INPUT ------------------------------------

# Simulation directory (location of run_issf_7.py on Linux system)
linuxSimulationPath = "/home/jhorne/projects/issf_case7"    # ABSOLUTE Linux path to simulation directory
# Virtual environment directory - Optional
vEnvDirectory = ".venv"     # EITHER the name of the vEnv directory OR None

# MOOSE setup
execPathStr = "/home/jhorne/projects/proteus/proteus-opt"   # ABSOLUTE Linux path to MOOSE executable
nTasks = 32                 # Number of cores to use

# Path to parameter data JSON file - Optional
inputJsonPathStr = None     # EITHER a path to the JSON file OR None

# Input parameters - ignored if using JSON input
coolantTemp=155             # degC
convectionHTC=150000        # W/m^2K
topSurfHeatFlux=2.5e7       # W/m^2
sideSurfHeatFlux=2e8        # W/m^2
protrusion=0.000           # m - the distance monoblock protrudes past neighbour
coolantPressure=5e6         # Pa

scale_therm_exp_CuCrZr=1.2
scale_therm_exp_Cu=1.1
scale_therm_exp_W=1.0

scale_therm_cond_CuCrZr=1.2
scale_therm_cond_Cu=1.0
scale_therm_cond_W=1.0

scale_spec_heat_CuCrZr=1.0
scale_spec_heat_Cu=1.05
scale_spec_heat_W=1.0

scale_density_CuCrZr=1.0
scale_density_Cu=1.0
scale_density_W=1.0

scale_youngs_CuCrZr=1.0
scale_youngs_Cu=1.0
scale_youngs_W=1.0

scale_poisson_CuCrZr=1.0
scale_poisson_Cu=1.0
scale_poisson_W=1.0

#------------------------- END -------------------------------------------
#-------------------------------------------------------------------------

if inputJsonPathStr is None:
    # Add parameters to dictionary
    params = {
        'coolantTemp':              coolantTemp,
        'convectionHTC':            convectionHTC,
        'topSurfHeatFlux':          topSurfHeatFlux,
        'sideSurfHeatFlux':         sideSurfHeatFlux,
        'coolantPressure':          coolantPressure,
        'protrusion':               protrusion,
        'scale_therm_exp_CuCrZr':   scale_therm_exp_CuCrZr,
        'scale_therm_exp_Cu':       scale_therm_exp_Cu,
        'scale_therm_exp_W':        scale_therm_exp_W,
        'scale_therm_cond_CuCrZr':  scale_therm_cond_CuCrZr,
        'scale_therm_cond_Cu':      scale_therm_cond_Cu,
        'scale_therm_cond_W':       scale_therm_cond_W,
        'scale_spec_heat_CuCrZr':   scale_spec_heat_CuCrZr,
        'scale_spec_heat_Cu':       scale_spec_heat_Cu,
        'scale_spec_heat_W':        scale_spec_heat_W,
        'scale_density_CuCrZr':     scale_density_CuCrZr,
        'scale_density_Cu':         scale_density_Cu,
        'scale_density_W':          scale_density_W,
        'scale_youngs_CuCrZr':      scale_youngs_CuCrZr,
        'scale_youngs_Cu':          scale_youngs_Cu,
        'scale_youngs_W':           scale_youngs_W,
        'scale_poisson_CuCrZr':     scale_poisson_CuCrZr,
        'scale_poisson_Cu':         scale_poisson_Cu,
        'scale_poisson_W':          scale_poisson_W
    }
    
    # Write dictionary to json
    with open(Path(DEFAULT_PARAM_JSON), 'w') as fp:
        json.dump(params, fp)
else:
    jsonPath = Path(inputJsonPathStr)
    if not jsonPath.exists():
        raise FileExistsError("Input parameter JSON file does not exist")

    jsonPath.replace(Path(DEFAULT_PARAM_JSON))

# Construct a command line command to:
#   1) Copy the input parameter JSON file to the simulation direction in the WSL filesystem
#   2) Activate the virtual environment in the simulation direction in the WSL filesystem
#   3) Run the Python script to setup and run MOOSE, all via WSL
#   4) Copy the results parameter JSON file back to Windows filesystem
command = f"wsl -e bash -lc \"cd {linuxSimulationPath} && cp $(wslpath \'{Path(DEFAULT_PARAM_JSON).resolve()}\') ./{DEFAULT_PARAM_JSON} && "

if not vEnvDirectory is None:
    command = command + f"source {vEnvDirectory}/bin/activate && "

command = command + f"python run_issf_7.py {execPathStr} {nTasks} {DEFAULT_PARAM_JSON} && "
command = command + f"cp {PARAM_OUTPUT_FILE} $(wslpath \'{Path(PARAM_OUTPUT_FILE).resolve()}\')\""

subprocess.run(command)

# Make results available
jsonPath = Path(PARAM_OUTPUT_FILE)
if jsonPath.exists():
    with open(jsonPath, 'r') as fp:
        results = json.load(fp)
else:
    raise FileExistsError("Results JSON file does not exist")

temp_max_W = results['temp_max_W']
stress_max_W = results['stress_max_W']
stress_max_CuCrZr = results['stress_max_CuCrZr']
strain_max_Cu = results['strain_max_Cu']
shear_stress_max_W_Cu = results['shear_stress_max_W_Cu']