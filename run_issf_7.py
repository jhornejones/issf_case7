from moose_wrapper import *
import json
import sys

INPUT_FILE = "monoblock.i"
SIM_OUTPUT_FILE = "monoblock_out.csv"
PARAM_OUTPUT_FILE = "results.json"
OTHER_DATA = ["data"]

if __name__ == '__main__':
    #-------------------------------------------------------------------------
    #------------------------- USER INPUT ------------------------------------
    
    # MOOSE setup
    execPathStr = "/home/jhorne/projects/proteus/proteus-opt"   # ABSOLUTE Linux path to MOOSE executable
    nTasks = 16                 # Number of cores to use
    
    # Input parameter reading - OPTIONAL
    inputJsonPathStr = None     # Either a path to a json file containing input parameters to update or None
    
    # Input parameters
    coolantTemp=155             # degC
    convectionHTC=150000        # W/m^2K
    topSurfHeatFlux=2e7         # W/m^2
    sideSurfHeatFlux=2e8        # W/m^2
    protrusion=0.000            # m - the distance monoblock protrudes past neighbour
    coolantPressure=5e6         # Pa

    scale_therm_exp_CuCrZr=1.0
    scale_therm_exp_Cu=1.0
    scale_therm_exp_W=1.0

    scale_therm_cond_CuCrZr=1.0
    scale_therm_cond_Cu=1.0
    scale_therm_cond_W=1.0

    scale_density_CuCrZr=1.0
    scale_density_Cu=1.0
    scale_density_W=1.0

    scale_youngs_CuCrZr=1.0
    scale_youngs_Cu=1.0
    scale_youngs_W=1.0

    scale_spec_heat_CuCrZr=1.0
    scale_spec_heat_Cu=1.0
    scale_spec_heat_W=1.0

    scale_poisson_CuCrZr=1.0
    scale_poisson_Cu=1.0
    scale_poisson_W=1.0
    
    #------------------------- END -------------------------------------------
    #-------------------------------------------------------------------------
    
    # Parse arguements if they exist (allows call from Windows)
    if len(sys.argv) > 1:
        if len(sys.argv) != 4:
            raise ValueError("Invalid number of command line arguments")
        else:
            execPathStr = sys.argv[1]
            
            if sys.argv[2].isnumeric():
                nTasks = int(sys.argv[2])
            
            inputJsonPathStr = sys.argv[3]
    
    if inputJsonPathStr is None:
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
    else:
        jsonPath = Path(inputJsonPathStr)
        if jsonPath.exists():
            with open(jsonPath, 'r') as fp:
                params = json.load(fp)
        else:
            raise FileExistsError("Input parameter JSON file does not exist")
    
    # Setup and run MOOSE simulation
    sim = MooseSim(
        inputFile=INPUT_FILE,
        outputFile=SIM_OUTPUT_FILE,
        execPathStr=execPathStr,
        otherData=OTHER_DATA
    )
    
    sim.updateInputFile(params)
    sim.runSimulation(nTasks) 
    results = sim.collectSteadyStateOutputs()
    
    # Make results available
    with open(Path(PARAM_OUTPUT_FILE), 'w') as fp:
        json.dump(results, fp)
    
    temp_max = results['temp_max']
    temp_avg = results['temp_avg']
    stress_max = results['stress_max']