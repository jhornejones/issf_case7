from moose_wrapper import *

INPUT_FILE = "monoblock.i"
OUTPUT_FILE = "monoblock_out.csv"
OTHER_DATA = ["data"]

if __name__ == '__main__':
    #-------------------------------------------------------------------------
    #------------------------- USER INPUT ------------------------------------
    
    # MOOSE setup
    execPathStr = "/home/jhorne/projects/proteus/proteus-opt"   # Path to MOOSE executable
    nTasks = 16                                                 # Number of cores to use
    
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
    
    params = {
        'coolantTemp':              coolantTemp,
        'convectionHTC':            convectionHTC,
        'topSurfHeatFlux':          topSurfHeatFlux,
        'sideSurfHeatFlux':         sideSurfHeatFlux,
        'coolantPressure':          coolantPressure,
        'protrusion':               protrusion,
        'scale_therm_cond_W':       scale_therm_cond_W,
        'scale_youngs_CuCrZr':      scale_youngs_CuCrZr,
    }
    
    sim = MooseSim(
        inputFile=INPUT_FILE,
        outputFile=OUTPUT_FILE,
        execPathStr=execPathStr,
        otherData=OTHER_DATA
    )
    
    sim.updateInputFile(params)
    
    sim.runSimulation(nTasks)
    
    results = sim.collectSteadyStateOutputs()
    
    print(results)