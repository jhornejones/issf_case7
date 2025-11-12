
from pathlib import Path
import sys
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

FILE_RAW_IN = "input_samples_"
FILE_RAW_IN_FINAL = "input_all_samples_"
FILE_RAW_IN_EXT = ".csv"
FILE_RAW_OUT = "output_samples_"
FILE_RAW_OUT_FINAL = "output_all_samples_"
FILE_RAW_OUT_EXT = ".csv"
FILE_SINGLE_IN = "input_single_"
FILE_SINGLE_OUT = "output_single_"
FILE_SINGLE_EXT = ".json"
FILE_FAIL = "failure_"
FILE_FAIL_BATCHES = "failure_batches_"
FILE_FAIL_CONV = "failure_convergence_"
FILE_FAIL_EXT = ".csv"
FILE_ERRORS = "errors_"
FILE_ERRORS_EXT = ".csv"

FILE_SLURM_SAMP = "samp.slurm"
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

N_BATCH_MIN = 10    # Minimum number of batches
N_RUN_MAX = 1E7     # Maximum number of model runs
SE_TOL = 1E-6

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
    parser.add_argument("--batch_no", help="Sampling batch number", type=int, default=0)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", type=int, default=N_CORES_DEFAULT)

    args = parser.parse_args()

    # Process restart if required
    if args.restart:
        files = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_IN}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_OUT}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_SINGLE_IN}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_SINGLE_OUT}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_FAIL}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_FAIL_BATCHES}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_FAIL_CONV}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_IN_FINAL}{args.id}" in f.name]
        files = files + [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_OUT_FINAL}{args.id}" in f.name]

        for f in files:
            Path.unlink(f)
        
        args.batch_no = 0

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
    samples.to_csv(sampleInputsFileName(args.id, args.batch_no))

    print(f"Generated {args.batch_size} new samples for case {args.id}. Total now {samples.index[-1]+1}")

    # Queue follow on jobs
    generateAndQueueExecAndEval(args)

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

def generateAndQueueExecAndEval(args):

    # Generate execution submission script
    generateExecutionSlurmScript(args)

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
    generateEvaluationSlurmScript(args, execJobId)

    # Queue evaluation job
    result = subprocess.run(
        ["sbatch", FILE_SLURM_EVAL],
        capture_output=True,
        text = True,
        shell = False
    )

    if result.stderr:
        raise RuntimeError("Evaluation queue submission failed")

def generateExecutionSlurmScript(args):
    with open(FILE_SLURM_EXEC, "w") as fp:
        # Fixed initial content
        fp.write("#!/bin/bash\n")
        fp.write("#SBATCH --job-name=issf7_exec\n")
        fp.write("#SBATCH --partition=cpu\n")
        fp.write("#SBATCH --time=1:00:00\n")
        fp.write("#SBATCH --nodes=1\n")
        fp.write("#SBATCH --cpus-per-task=1\n")
        fp.write(f"#SBATCH --mem-per-cpu=4G\n")
        
        # Args dependent slurm input
        fp.write(f"#SBATCH --ntasks-per-node={args.n_cores}\n")
        fp.write(f"#SBATCH --array=0-{args.batch_size - 1}\n")

        # Commands
        fp.write("source .venv/bin/activate\n")

        runCom = "python run_monte_carlo.py --exec_type execution "
        runCom += f"--id {args.id} "
        runCom += f"--batch_no {args.batch_no} "
        runCom += f"--n_cores {args.n_cores} "
        runCom += f"--exec_path {args.exec_path} "
        runCom += f"--i_array $SLURM_ARRAY_TASK_ID "

        runCom += "\n"

        fp.write(runCom)

def generateEvaluationSlurmScript(args, execJobId):
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
        runCom += f"--batch_no {args.batch_no} "
        runCom += f"--batch_size {args.batch_size} "
        runCom += f"--n_cores {args.n_cores} "
        runCom += f"--exec_path {args.exec_path} "

        runCom += "\n"

        fp.write(runCom)

def sampleInputsFileName(id, batchNo) -> str:
    return f"{FILE_RAW_IN}{id}_{batchNo:05d}{FILE_RAW_IN_EXT}"

def sampleOutputsFileName(id, batchNo) -> str:
    return f"{FILE_RAW_OUT}{id}_{batchNo:05d}{FILE_RAW_OUT_EXT}"

def allInputsFileName(id) -> str:
    return f"{FILE_RAW_IN_FINAL}{id}{FILE_RAW_IN_EXT}"

def allOutputsFileName(id) -> str:
    return f"{FILE_RAW_OUT_FINAL}{id}{FILE_RAW_OUT_EXT}"

def singleInputFileName(id, batchIndex) -> str:
    return f"{FILE_SINGLE_IN}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def singleOutputFileName(id, batchIndex) -> str:
    return f"{FILE_SINGLE_OUT}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def failuresFileName(id) -> str:
    return f"{FILE_FAIL}{id}{FILE_FAIL_EXT}"

def failuresBatchesFileName(id) -> str:
    return f"{FILE_FAIL_BATCHES}{id}{FILE_FAIL_EXT}"

def failuresConvergenceFileName(id) -> str:
    return f"{FILE_FAIL_CONV}{id}{FILE_FAIL_EXT}"

def errorsFileName(id) -> str:
    return f"{FILE_ERRORS}{id}{FILE_ERRORS_EXT}"

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

    subprocess.run(
        command,
        shell = False
    )

    print("Completed execution", flush=True)

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
    allData = {}
    for i in range(args.batch_size):
        fPath = Path(singleOutputFileName(args.id, i))

        if fPath.exists():
            with open(fPath, "r") as fp:
                allData.update({i: json.load(fp)})
        else:
            print(f"WARNING: model run {i} failed")
    
    combinedData = pd.DataFrame.from_dict(allData, orient='index')
    combinedData.to_csv(sampleOutputsFileName(args.id, args.batch_no))

    # Delete individual output files
    singleInputFiles = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_SINGLE_IN}{args.id}" in f.name]
    singleOutputFiles = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_SINGLE_OUT}{args.id}" in f.name]

    for f in singleInputFiles+singleOutputFiles:
        f.unlink()

    print("Combined all model run outputs", flush=True)

    # Compute and save out failures
    combinedData["stress_max_W_cons"] = combinedData["stress_max_W"].copy()
    
    failures = combinedData.sub(pd.Series(FAILURE_CRIT), axis=1)
    failures = failures.map(lambda x: 1 if x >= 0 else 0)
    
    probFail = failures.mean(axis=0)

    probFail["batch_size"] = len(combinedData.index)
    probFail.name = args.batch_no

    ## Append batch failures to file
    if Path(failuresBatchesFileName(args.id)).exists():
        allFail = pd.read_csv(failuresBatchesFileName(args.id), index_col=0)
        if args.batch_no in allFail.index:
            allFail.loc[args.batch_no] = probFail
        else:
            allFail = pd.concat([allFail,pd.DataFrame(probFail).T])
    else:
        allFail = pd.DataFrame(probFail).T
    
    allFail.to_csv(failuresBatchesFileName(args.id))
    
    ## Compute overall failure and save
    combFail = allFail.copy()
    combFail.loc[:, combFail.columns != "batch_size"] = combFail.loc[:, combFail.columns != "batch_size"].multiply(combFail["batch_size"], axis=0)
    
    combFail = pd.DataFrame(combFail.sum(axis=0)).T
    combFail.rename(columns={"batch_size": "run_count"}, inplace=True)

    combFail.loc[:, combFail.columns != "run_count"] = combFail.loc[:, combFail.columns != "run_count"].divide(combFail["run_count"], axis=0)
    
    combFail.to_csv(failuresFileName(args.id))

    if Path(failuresConvergenceFileName(args.id)).exists():
        convFail = pd.read_csv(failuresConvergenceFileName(args.id), index_col=0)
        if args.batch_no in convFail.index:
            convFail.loc[args.batch_no] = combFail.iloc[0]
        else:
            convFail = pd.concat([convFail,combFail])
    else:
        convFail = combFail
    
    convFail.to_csv(failuresConvergenceFileName(args.id))
    
    print("Computed failure probabilities", flush=True)

    generateFinalSamplesFiles(args)

    # Determine stopping status
    if combFail["run_count"].iloc[0] >= N_RUN_MAX:
        print("Hit run count limit")
    else:
        if args.batch_no < N_BATCH_MIN-1:
            print("Insufficient number of batches", flush=True)

            errors = computeErrors(allFail)
            errors.to_csv(errorsFileName(args.id))

            generateAndQueueSampling(args)
        else:
            errors = computeErrors(allFail)
            errors.to_csv(errorsFileName(args.id))

def computeErrors(allFail) -> pd.DataFrame:
    sigma = allFail.drop(columns="batch_size").std(axis=0)
    se = sigma / m.sqrt(len(allFail.index))
    re = se / allFail.drop(columns="batch_size").mean(axis=0)

    tmp = {
        "standard error": se,
        "relative error": re
    }
    
    return pd.concat(tmp, axis=1).T

def generateFinalSamplesFiles(args):
    inputFiles = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_IN}{args.id}" in f.name]
    outputFiles = [f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_RAW_OUT}{args.id}" in f.name]

    if len(inputFiles) != len(outputFiles):
        raise FileExistsError("Input and output files not matching")
    
    allIn = []
    allOut = []

    for inp, out in zip(inputFiles, outputFiles):
        inNum = int(inp.stem.split("_")[-1])
        outNum = int(out.stem.split("_")[-1])

        if inNum != outNum:
            raise FileExistsError("Input and output files not matching")
        
        dataIn = pd.read_csv(inp, index_col=0)
        dataOut = pd.read_csv(out, index_col=0)

        dataIn = dataIn.loc[dataOut.index]      # Remove samples where simulation failed

        dataIn.reset_index(drop=True, inplace=True)
        dataOut.reset_index(drop=True, inplace=True)

        allIn.append(dataIn)
        allOut.append(dataOut)

    allDataIn = pd.concat(allIn, ignore_index=True)
    allDataOut = pd.concat(allOut, ignore_index=True)

    allDataIn.to_csv(allInputsFileName(args))
    allDataOut.to_csv(allOutputsFileName(args))

    print("Generated final samples files", flush=True)

def generateAndQueueSampling(args):

    # Generate slurm file
    generateSamplingSlurmScript(args)

    # Queue job
    result = subprocess.run(
        ["sbatch", FILE_SLURM_SAMP],
        capture_output=True,
        text = True,
        shell = False
    )

    if result.stderr:
        raise RuntimeError("Sampling queue submission failed")
    
    print("Queued sampling job", flush=True)

def generateSamplingSlurmScript(args):
    with open(FILE_SLURM_SAMP, "w") as fp:
        # Fixed initial content
        fp.write("#!/bin/bash\n")
        fp.write("#SBATCH --job-name=issf7_samp\n")
        fp.write("#SBATCH --partition=cpu\n")
        fp.write("#SBATCH --time=0:10:00\n")
        fp.write("#SBATCH --nodes=1\n")
        fp.write("#SBATCH --cpus-per-task=1\n")
        fp.write("#SBATCH --ntasks-per-node=1\n")

        # Commands
        fp.write("source .venv/bin/activate\n")

        runCom = "python run_monte_carlo.py --exec_type sampling "
        runCom += f"--id {args.id} "
        runCom += f"--batch_no {args.batch_no+1} "
        runCom += f"--batch_size {args.batch_size} "
        runCom += f"--n_cores {args.n_cores} "
        runCom += f"--exec_path {args.exec_path} "

        runCom += "\n"

        fp.write(runCom)

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