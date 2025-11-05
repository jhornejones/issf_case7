
from pathlib import Path
import argparse
import json
import subprocess

import math as m

from scipy import stats
import pandas as pd

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

N_CORES_DEFAULT = 6

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
    print("Running sampling")

    parser.add_argument("--id", help="Subcase id", default="a", choices=["a", "b", "c", "bandc"])
    parser.add_argument("--restart", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", default=N_CORES_DEFAULT)

    args = parser.parse_args()

    batchSize = 1000
    batchSize = 10          # Remove after testing

    # Process restart if required
    if args.restart:
        files = [f for f in Path.cwd().iterdir() if f.is_file() and (f"{FILE_IN}{args.id}" in f.name or f"{FILE_OUT}{args.id}" in f.name or f"{FILE_IN_SINGLE}{args.id}" in f.name or f"{FILE_OUT_SINGLE}{args.id}" in f.name)]
        for f in files:
            Path.unlink(f)

    # Run sampling
    if args.id == "a":
        samples = generateSamplesCaseA(batchSize)
    elif args.id == "b":
        samples = generateSamplesCaseB(batchSize)
    elif args.id == "c":
        samples = generateSamplesCaseC(batchSize)
    elif args.id == "bandc":
        samples = generateSamplesCaseC(batchSize)

    # Save sampling data
    files = sorted([f for f in Path.cwd().iterdir() if f.is_file() and f"{FILE_IN}{args.id}" in f.name], key=lambda x: x.stem)
    if files:
        temp = pd.read_csv(files[-1], index_col=0)
        indStart = temp.index[-1] + 1
        samples.index = list(range(indStart,indStart+batchSize,1))

    samples.to_csv(sampleInputsFileName(args.id, len(files)))

    print(f"Generated {batchSize} new samples for case {args.id}. Total now {samples.index[-1]+1}")

    # Queue exec job
    # ADD HERE

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

def sampleInputsFileName(id, batchNo) -> str:
    return f"{FILE_IN}{id}_{batchNo:05d}{FILE_IN_EXT}"

def singleInputFileName(id, batchIndex) -> str:
    return f"{FILE_IN_SINGLE}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def singleOutputFileName(id, batchIndex) -> str:
    return f"{FILE_OUT_SINGLE}{id}_{batchIndex:05d}{FILE_SINGLE_EXT}"

def runExecution(argv):
    print("Running execution")

    # Parse inputs
    parser.add_argument("--id", help="Subcase id", default="a", choices=["a", "b", "c", "bandc"])
    parser.add_argument("--batch_no", help="Sample batch number", default=0)
    parser.add_argument("--exec_path", help="Path to MOOSE executable")
    parser.add_argument("--n_cores", help="Number of cores per MOOSE instance", default=N_CORES_DEFAULT)
    parser.add_argument("--i_array", help="SLURM_ARRAY_TASK_ID variable", default=0)

    args = parser.parse_args()

    # Generate MOOSE input for given sample
    samples = pd.read_csv(sampleInputsFileName(args.id, args.batch_no), index_col=0)
    sample = samples.iloc[args.i_array].to_dict()
    
    with open(singleInputFileName(args.id, args.i_array), "w") as fp:
        json.dump(sample, fp)

    # Run MOOSE
    command = f"python run_issf_7.py {args.exec_path} {args.n_cores} {singleInputFileName(args.id, args.i_array)} {singleOutputFileName(args.id, args.i_array)}"
    subprocess.run(command, shell=True)

def runEvaluation(argv):
    print("Running evaluation")

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