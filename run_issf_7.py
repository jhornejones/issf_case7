from moose_wrapper import *
import json
import sys

PARENT_INPUT_FILE = "monoblock_plastic.i"
MULTIAPP_INPUT_FILES = ["monoblock_thermal.i"]
SIM_OUTPUT_FILE = "monoblock_out.csv"
PARAM_OUTPUT_FILE = "results.json"
OTHER_DATA = ["data"]

if __name__ == '__main__':

    print("run_issf_7 called", flush=True)
    #-------------------------------------------------------------------------
    #------------------------- USER INPUT ------------------------------------
    
    # MOOSE setup
    execPathStr = "/home/jhorne/projects/proteus/proteus-opt"   # ABSOLUTE Linux path to MOOSE executable
    nTasks = 6                 # Number of cores to use
    
    # Input parameter reading - OPTIONAL
    inputJsonPathStr = None     # Either a path to a json file containing input parameters to update or None  
    
    # Input parameters
    coolantTemp=155             # degC
    convectionHTC=150000        # W/m^2K

    topSurfHeatFlux=2.5e7       # W/m^2

    sideSurfHeatFlux=2e8        # W/m^2
    protrusion=0.000            # m - the distance monoblock protrudes past neighbour

    scale_therm_exp_Cu=1.0

    scale_therm_cond_CuCrZr=1.0
    scale_therm_cond_W=1.0

    scale_youngs_CuCrZr=1.0
    scale_youngs_W=1.0

    scale_poisson_Cu=0.6
    
    #------------------------- END -------------------------------------------
    #-------------------------------------------------------------------------
    
    outputJsonPathStr = PARAM_OUTPUT_FILE   # Default value

    # Parse arguments if they exist (allows call from other script)
    if len(sys.argv) > 1:
        if len(sys.argv) == 4 or len(sys.argv) == 5:
            execPathStr = sys.argv[1]
            
            if sys.argv[2].isnumeric():
                nTasks = int(sys.argv[2])
            
            inputJsonPathStr = sys.argv[3]

            if len(sys.argv) == 5:
                outputJsonPathStr = sys.argv[4]
        else:
            raise ValueError("Invalid number of command line arguments")
    
    if inputJsonPathStr is None:
        params = {
            'coolantTemp':              coolantTemp,
            'convectionHTC':            convectionHTC,
            'topSurfHeatFlux':          topSurfHeatFlux,
            'sideSurfHeatFlux':         sideSurfHeatFlux,
            'protrusion':               protrusion,
            'scale_therm_exp_Cu':       scale_therm_exp_Cu,
            'scale_therm_cond_CuCrZr':  scale_therm_cond_CuCrZr,
            'scale_therm_cond_W':       scale_therm_cond_W,
            'scale_youngs_CuCrZr':      scale_youngs_CuCrZr,
            'scale_youngs_W':           scale_youngs_W,
            'scale_poisson_Cu':         scale_poisson_Cu,
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
        inputFile=PARENT_INPUT_FILE,
        outputFile=SIM_OUTPUT_FILE,
        execPathStr=execPathStr,
        multiAppInputFiles=MULTIAPP_INPUT_FILES,
        otherData=OTHER_DATA
    )
    
    sim.updateInputFile(params)
    sim.runSimulation(nTasks) 
    results = sim.collectSteadyStateOutputs()
    
    # Make results available
    with open(Path(outputJsonPathStr), 'w') as fp:
        json.dump(results, fp)
    
    temp_max_W = results['temp_max_W']
    stress_max_W = results['stress_max_W']
    stress_max_CuCrZr = results['stress_max_CuCrZr']
    strain_max_Cu = results['strain_max_Cu']
    shear_stress_max_W_Cu = results['shear_stress_max_W_Cu']

    print("run_issf_7 completed", flush=True)