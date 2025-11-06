
from pathlib import Path
import argparse
import json
import subprocess

import math as m

from scipy import stats
import pandas as pd
import numpy as np

ARG_SAMPLE = "sampling"
ARG_EXEC = "execution"
ARG_EVAL = "evaluation"

FILE_IN = "input_samples_"
FILE_IN_EXT = ".csv"
FILE_OUT = "output_samples_"
FILE_OUT_EXT = ".csv"
FILE_IN_SINGLE = "input_single_"
FILE_OUT_SINGLE = "output_single_"
FILE_SINGLE_EXT = ".json"

FILE_SLURM_EXEC = "exec.slurm"
FILE_SLURM_EVAL = "eval.slurm"

N_CORES_DEFAULT = 6

FAILURE_CRIT = {
    "shear_stress_max_W_Cu": 39E6,
    "strain_max_Cu": 0.1,
    "stress_max_CuCrZr": 400E6,
    "stress_max_W": 750E6,
    "temp_max_W": 3422,
    "stress_max_W_cons": 300E6
}

"""
Read run type (sampling, execution, or evaluation)

Sampling
    Read case id from argv
    Read whether to restart sampling from argv
        If restarting remove all existing files (in and out)
    Generate new samples
    Save new samples
    Schedule execution job
    Schedule evaluation job, depending on execution job

Execution
    Read case id from argv
    Read sample file name from argv
    Read sample based on slurm array number
    Execute
    Save out

Evaluation
    Read case id from argv
    Read in all output files and combine
    Write out combined results
    Delete individual results
    Compute probabilities of failure
    Write out reliability results
    Assess stopping criteria
        If not converged, schedule additional sampling job
"""

def runSampling(parser):
    print("Running sampling", flush=True)

    parser.add_argument("--id", help="Subcase id", default="a", choices=["a", "b", "c", "bandc"])
    parser.add_argument("--restart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--batch_size", help="Sampling batch size", type=int, default=1000)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", type=int, default=N_CORES_DEFAULT)

    args = parser.parse_args()

    # Process restart if required
    if args.restart:
        files = [f for f in Path.cwd().iterdir() if f.is_file() and (f"{FILE_IN}{args.id}" in f.name or f"{FILE_OUT}{args.id}" in f.name or f"{FILE_IN_SINGLE}{args.id}" in f.name or f"{FILE_OUT_SINGLE}{args.id}" in f.name)]
        for f in files:
            Path.unlink(f)

    # Run sampling
    if args.id == "a":
        samples = generateSamplesCaseA(args.batch_size)
    elif args.id == "b":
        samples = generateSamplesCaseB(args.batch_size)
    elif args.id == "c":
        samples = generateSamplesCaseC(args.batch_size)
    elif args.id == "bandc":
        samples = generateSamplesCaseC(args.batch_size)

    # Save sampling data
    files = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_IN}{args.id}" in f.name]

    batchNo = len(files)
    samples.to_csv(sampleInputsFileName(args.id, batchNo))

    print(f"Generated {args.batch_size} new samples for case {args.id}. Total now {samples.index[-1]+1}")

    # Queue follow on jobs
    generateAndQueueExecAndEval(args, batchNo)

    print("Queued execution and evaluation jobs", flush=True)

def generateSamplesCaseA(batchSize: int) -> pd.DataFrame:

    samples_dict = {
        "topSurfHeatFlux":          stats.norm.rvs(loc=2.0E7, scale=m.sqrt(4.0E12), size=batchSize),
        "coolantTemp":              stats.norm.rvs(loc=150.0, scale=m.sqrt(10.0), size=batchSize),
        "convectionHTC":            stats.norm.rvs(loc=1.5E5, scale=m.sqrt(1.0E8), size=batchSize),
        "scale_therm_exp_Cu":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_CuCrZr":  stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_W":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_CuCrZr":      stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_W":           stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_poisson_Cu":         stats.truncnorm.rvs(a=-4.0, b=4.0, loc=1.0, scale=m.sqrt(0.01), size=batchSize)
    }

    samples = pd.DataFrame(samples_dict)
    return samples

def generateSamplesCaseB(batchSize: int) -> pd.DataFrame:

    samples_dict = {
        "topSurfHeatFlux":          stats.weibull_min.rvs(c=10.0, scale=2.1E7, size=batchSize),
        "coolantTemp":              stats.weibull_min.rvs(c=30.0, scale=150.0, size=batchSize),
        "convectionHTC":            stats.norm.rvs(loc=1.5E5, scale=m.sqrt(1.0E8), size=batchSize),
        "scale_therm_exp_Cu":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_CuCrZr":  stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_W":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_CuCrZr":      stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_W":           stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_poisson_Cu":         stats.truncnorm.rvs(a=-4.0, b=4.0, loc=1.0, scale=m.sqrt(0.01), size=batchSize)
    }

    samples = pd.DataFrame(samples_dict)
    return samples

def generateSamplesCaseC(batchSize: int) -> pd.DataFrame:

    samples_dict = {
        "topSurfHeatFlux":          stats.weibull_min.rvs(c=10.0, scale=2.1E7, size=batchSize),
        "sideSurfHeatFlux":         stats.norm.rvs(loc=2.0E8, scale=m.sqrt(4.0E14), size=batchSize),
        "protrusion":               stats.truncnorm.rvs(a=-4.0, b=4.0, loc=-0.01, scale=m.sqrt(9.0E-6), size=batchSize),
        "coolantTemp":              stats.weibull_min.rvs(c=30.0, scale=150.0, size=batchSize),
        "convectionHTC":            stats.norm.rvs(loc=1.5E5, scale=m.sqrt(1.0E8), size=batchSize),
        "scale_therm_exp_Cu":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_CuCrZr":  stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_therm_cond_W":       stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_CuCrZr":      stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_youngs_W":           stats.norm.rvs(loc=1.0, scale=m.sqrt(0.0156), size=batchSize),
        "scale_poisson_Cu":         stats.truncnorm.rvs(a=-4.0, b=4.0, loc=1.0, scale=m.sqrt(0.01), size=batchSize)
    }

    samples = pd.DataFrame(samples_dict)
    return samples

def generateAndQueueExecAndEval(args, batchNo):

    # Generate execution submission script
    generateExecutionSlurmScript(args, batchNo)

    # Queue execution job
    result = subprocess.run(
        ["sbatch", FILE_SLURM_EXEC],
        capture_output=True,
        text = True,
        shell = False
    )

    if result.stderr:
        raise RuntimeError("Execution queue submission failed")
    
    execJobId = result.stdout.split(" ")[-1]

    # Generate evaluation submission script
    generateEvaluationSlurmScript(args, batchNo, execJobId)

    # Queue evaluation job
    result = subprocess.run(
        ["sbatch", FILE_SLURM_EVAL],
        capture_output=True,
        text = True,
        shell = False
    )

    if result.stderr:
        raise RuntimeError("Evaluation queue submission failed")

def generateExecutionSlurmScript(args, batchNo):
    with open(FILE_SLURM_EXEC, "w") as fp:
        # Fixed initial content
        fp.write("#!/bin/bash\n")
        fp.write("#SBATCH --job-name=issf7_exec\n")
        fp.write("#SBATCH --partition=cpu\n")
        fp.write("#SBATCH --time=1:00:00\n")
        fp.write("#SBATCH --nodes=1\n")
        fp.write("#SBATCH --cpus-per-task=1\n")
        
        # Args dependent slurm input
        fp.write(f"#SBATCH --ntasks-per-node={args.n_cores}\n")
        fp.write(f"#SBATCH --array=0-{args.batch_size - 1}\n")

        # Commands
        fp.write("source .venv/bin/activate\n")

        runCom = "python run_monte_carlo.py --exec_type execution "
        runCom += f"--id {args.id} "
        runCom += f"--batch_no {batchNo} "
        runCom += f"--n_cores {args.n_cores} "
        runCom += f"--exec_path {args.exec_path} "
        runCom += f"--i_array $SLURM_ARRAY_TASK_ID "

        runCom += "\n"

        fp.write(runCom)

def generateEvaluationSlurmScript(args, batchNo, execJobId):
    with open(FILE_SLURM_EVAL, "w") as fp:
        # Fixed initial content
        fp.write("#!/bin/bash\n")
        fp.write("#SBATCH --job-name=issf7_eval\n")
        fp.write("#SBATCH --partition=cpu\n")
        fp.write("#SBATCH --time=0:10:00\n")
        fp.write("#SBATCH --nodes=1\n")
        fp.write("#SBATCH --cpus-per-task=1\n")
        fp.write("#SBATCH --ntasks-per-node=1\n")

        # Dependency on exec job
        fp.write(f"#SBATCH --dependency=afterany:{execJobId}\n")

        # Commands
        fp.write("source .venv/bin/activate\n")

        runCom = "python run_monte_carlo.py --exec_type evaluation "
        runCom += f"--id {args.id} "
        runCom += f"--batch_no {batchNo} "
        runCom += f"--batch_size {args.batch_size} "
        runCom += f"--n_cores {args.n_cores} "
        runCom += f"--exec_path {args.exec_path} "

        runCom += "\n"

        fp.write(runCom)

def sampleInputsFileName(id, batchNo) -> str:
    return f"{FILE_IN}{id}_{batchNo:05d}{FILE_IN_EXT}"

def sampleOutputsFileName(id, batchNo) -> str:
    return f"{FILE_OUT}{id}_{batchNo:05d}{FILE_OUT_EXT}"

def singleInputFileName(id, batchIndex) -> str:
    return f"{FILE_IN_SINGLE}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def singleOutputFileName(id, batchIndex) -> str:
    return f"{FILE_OUT_SINGLE}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def runExecution(argv):
    print("Running execution", flush=True)

    # Parse inputs
    parser.add_argument("--id", help="Subcase id", default="a", choices=["a", "b", "c", "bandc"])
    parser.add_argument("--batch_no", help="Sample batch number", type=int, default=0)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", type=int, default=N_CORES_DEFAULT)
    parser.add_argument("--i_array", help="SLURM_ARRAY_TASK_ID variable", type=int, default=0)

    args = parser.parse_args()

    # Generate MOOSE input for given sample
    samples = pd.read_csv(sampleInputsFileName(args.id, args.batch_no), index_col=0)
    sample = samples.iloc[args.i_array].to_dict()
    
    with open(singleInputFileName(args.id, args.i_array), "w") as fp:
        json.dump(sample, fp)

    # Run MOOSE via ISSF run script
    command = [
        "python",
        "run_issf_7.py",
        args.exec_path,
        str(args.n_cores),
        singleInputFileName(args.id, args.i_array),
        singleOutputFileName(args.id, args.i_array)
    ]
    
    subprocess.run(command, shell=False)

def runEvaluation(argv):
    print("Running evaluation", flush=True)

    # Parse inputs
    parser.add_argument("--id", help="Subcase id", default="a", choices=["a", "b", "c", "bandc"])
    parser.add_argument("--batch_no", help="Sample batch number", type=int, default=0)
    parser.add_argument("--batch_size", help="Sampling batch size", type=int)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", type=int, default=N_CORES_DEFAULT)
    
    args = parser.parse_args()

    # Read in output files and combine
    singleOutputFiles = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_OUT_SINGLE}{args.id}" in f.name]
    if len(singleOutputFiles) != args.batch_size:
        raise ValueError("Some execution jobs failed")
    
    allData = {}
    for i in range(args.batch_size):
        with open(singleOutputFileName(args.id, i), "r") as fp:
            allData.update({i: json.load(fp)})
    
    combinedData = pd.DataFrame.from_dict(allData, orient='index')
    combinedData.to_csv(sampleOutputsFileName(args.id, args.batch_no))

    # Compute failures
    combinedData["stress_max_W_cons"] = combinedData["stress_max_W"].copy()
    
    combinedData.sub(pd.Series(FAILURE_CRIT), axis=0)
    failures = pd.DataFrame(np.zeros(combinedData.to_numpy().shape))
    

    print(failures)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('--exec_type', help='Execution type', required=True, choices=[ARG_SAMPLE, ARG_EXEC, ARG_EVAL])

    args, _ = parser.parse_known_args()

    if args.exec_type == ARG_SAMPLE:
        runSampling(parser)
    elif args.exec_type == ARG_EXEC:
        runExecution(parser)
    elif args.exec_type == ARG_EVAL:
        runEvaluation(parser)
    else:
        raise ValueError("Invalid execution type argument")