import pandas as pd
from scipy.stats import qmc
from pathlib import Path
import json
import subprocess

DEFAULT_PARAM_JSON = "params.json"
PARAM_OUTPUT_FILE = "results.json"

DOE_FILE = "doe.csv"
DOE_RES_FILE = "doeResults.csv"

execPathStr = "/home/jhorne/projects/proteus/proteus-opt"   # ABSOLUTE Linux path to MOOSE executable
nTasks = 32                 # Number of cores to use

nDOE = 100

coolantTemp=[125, 175]             # degC
convectionHTC=[125000, 175000]        # W/m^2K
topSurfHeatFlux=[1.5e7, 2.5e7]         # W/m^2
#sideSurfHeatFlux=[1.5e8, 2.5e8]        # W/m^2
#protrusion=[-0.009, 0.001]           # m - the distance monoblock protrudes past neighbour
coolantPressure=[4.5e6, 5.5e6]         # Pa

scale_therm_exp_CuCrZr=[0.75, 1.25]
scale_therm_exp_Cu=[0.8, 1.2]
scale_therm_exp_W=[0.75, 1.25]

scale_therm_cond_CuCrZr=[0.75, 1.25]
scale_therm_cond_Cu=[0.75, 1.25]
scale_therm_cond_W=[0.75, 1.25]

scale_density_CuCrZr=[0.75, 1.25]
scale_density_Cu=[0.75, 1.25]
scale_density_W=[0.75, 1.25]

scale_youngs_CuCrZr=[0.75, 1.25]
scale_youngs_Cu=[0.75, 1.25]
scale_youngs_W=[0.75, 1.25]

scale_spec_heat_CuCrZr=[0.75, 1.25]
scale_spec_heat_Cu=[0.75, 1.25]
scale_spec_heat_W=[0.75, 1.25]

scale_poisson_CuCrZr=[0.75, 1.25]
scale_poisson_Cu=[0.75, 1.25]
scale_poisson_W=[0.75, 1.25]

sampler = qmc.LatinHypercube(d=22)
sample = sampler.random(n=nDOE)

lb = [
    coolantTemp[0],
    convectionHTC[0],
    topSurfHeatFlux[0],
    coolantPressure[0],
    scale_therm_exp_CuCrZr[0],
    scale_therm_exp_Cu[0],
    scale_therm_exp_W[0],
    scale_therm_cond_CuCrZr[0],
    scale_therm_cond_Cu[0],
    scale_therm_cond_W[0],
    scale_density_CuCrZr[0],
    scale_density_Cu[0],
    scale_density_W[0],
    scale_youngs_CuCrZr[0],
    scale_youngs_Cu[0],
    scale_youngs_W[0],
    scale_spec_heat_CuCrZr[0],
    scale_spec_heat_Cu[0],
    scale_spec_heat_W[0],
    scale_poisson_CuCrZr[0],
    scale_poisson_Cu[0],
    scale_poisson_W[0],
]

ub = [
    coolantTemp[1],
    convectionHTC[1],
    topSurfHeatFlux[1],
    coolantPressure[1],
    scale_therm_exp_CuCrZr[1],
    scale_therm_exp_Cu[1],
    scale_therm_exp_W[1],
    scale_therm_cond_CuCrZr[1],
    scale_therm_cond_Cu[1],
    scale_therm_cond_W[1],
    scale_density_CuCrZr[1],
    scale_density_Cu[1],
    scale_density_W[1],
    scale_youngs_CuCrZr[1],
    scale_youngs_Cu[1],
    scale_youngs_W[1],
    scale_spec_heat_CuCrZr[1],
    scale_spec_heat_Cu[1],
    scale_spec_heat_W[1],
    scale_poisson_CuCrZr[1],
    scale_poisson_Cu[1],
    scale_poisson_W[1],
]

sample_scaled = qmc.scale(sample, lb, ub)

doe = pd.DataFrame(
    sample_scaled,
    columns=[
        'coolantTemp',
        'convectionHTC',
        'topSurfHeatFlux',
        'coolantPressure',
        'scale_therm_exp_CuCrZr',
        'scale_therm_exp_Cu',
        'scale_therm_exp_W',
        'scale_therm_cond_CuCrZr',
        'scale_therm_cond_Cu',
        'scale_therm_cond_W',
        'scale_spec_heat_CuCrZr',
        'scale_spec_heat_Cu',
        'scale_spec_heat_W',
        'scale_density_CuCrZr',
        'scale_density_Cu',
        'scale_density_W',
        'scale_youngs_CuCrZr',
        'scale_youngs_Cu',
        'scale_youngs_W',
        'scale_poisson_CuCrZr',
        'scale_poisson_Cu',
        'scale_poisson_W'
    ]
)

allResultDicts = []
command = f"python run_issf_7.py {execPathStr} {nTasks} {DEFAULT_PARAM_JSON}"
args = [
    'python',
    "run_issf_7.py",
    execPathStr,
    str(nTasks),
    DEFAULT_PARAM_JSON
]

for i in range(nDOE):
    params = doe.iloc[i,:].to_dict()
    with open(Path(DEFAULT_PARAM_JSON), 'w') as fp:
        json.dump(params, fp)
        
    try:
        subprocess.run(
            args,
            shell=False,
            cwd=Path.cwd(),
        )
        
        jsonPath = Path(PARAM_OUTPUT_FILE)
        if jsonPath.exists():
            with open(jsonPath, 'r') as fp:
                resultDict = json.load(fp)
        else:
            raise FileExistsError("Results JSON file does not exist")
    except:
        print(f"Simulation {i} failed")
        resultDict = {
            'temp_max_W': 0.0,
            'stress_max_W': 0.0,
            'stress_max_CuCrZr': 0.0,
            'strain_max_Cu': 0.0,
            'shear_stress_max_W_Cu': 0.0
        }        
    
    allResultDicts.append(resultDict)
        
results = pd.DataFrame(allResultDicts)

doe.to_csv(DOE_FILE)
results.to_csv(DOE_RES_FILE)