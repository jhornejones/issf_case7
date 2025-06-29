from pathlib import Path
import uuid
import shutil
import subprocess
import pandas as pd

INPUT_FILE = "monoblock.i"
OUTPUT_FILE = "monoblock_out.csv"

MOOSE_COMMENT = '#'
PARAM_START = "*_*START*_*"
PARAM_END = "*_*END*_*"

def directoryGenerator() -> Path:
    # Create randomly named folder for current simulation
    dirName = str(uuid.uuid4())
    Path(dirName).mkdir(parents=True, exist_ok=True)
    dirPath = Path(dirName)

    # Copy MOOSE input file and data folder
    dataDir = "data"
    inputFile = INPUT_FILE

    dataDstPath = dirPath / dataDir
    shutil.copytree("data", dataDstPath, dirs_exist_ok=True)
    inputDstPath = dirPath / inputFile
    shutil.copyfile(inputFile, inputDstPath)

    return dirPath

# Content derived from mooseherder
def updateInputFile(dirPath: Path, params: dict):
    # Read in input file
    inputPath = dirPath / INPUT_FILE

    with open(inputPath, "r", encoding="utf-8") as in_file:
        inputLines = in_file.readlines()
    
    # Find editable parameters
    foundStart = False
    setVarDict = {}

    for idx, line in enumerate(inputLines):
        if not foundStart:
            if PARAM_START in line:
                foundStart = True
        else:
            if PARAM_END in line:
                break
            else:
                line = line.strip()
                if line.startswith(MOOSE_COMMENT) or not line:
                    continue
                else:
                    variable, value, comment = splitMOOSEInputLine(line)
                    setVarDict.update({variable: (idx, comment)})
    
    # Check for invalid parameters
    if not set(params.keys()).issubset(setVarDict.keys()):
        invalidKeys = list(set(params.keys()) - set(setVarDict.keys()))
        raise KeyError(f"Invalid parameters: {invalidKeys}")
    
    # Update parameters
    for variable, contents in setVarDict.items():
        if variable in params.keys():
            value = params[variable]
            idx = contents[0]
            comment = contents[1]
            line = f"{variable} = {value} {MOOSE_COMMENT} {comment}\n"
            
            inputLines[idx] = line
    
    # Write out updated input file
    with open(inputPath, "w", encoding="utf-8") as out_file:
        out_file.writelines(inputLines)

# Content derived from mooseherder
def splitMOOSEInputLine(strippedLine: str) -> tuple:
    commentSplit = strippedLine.split(MOOSE_COMMENT, 1)     # Split by first comment
    equalsSplit = commentSplit[0].strip().split("=", 1)     # Split by first =

    variable = equalsSplit[0].strip()
    valueStr = equalsSplit[1].strip()

    try:
        value = float(valueStr)
        if value.is_integer():
            value = int(value)
    except ValueError:
        value = valueStr
    
    if len(commentSplit) > 1:
        comment = commentSplit[1]
    else:
        comment = ""

    return (variable, value, comment)

def runSimulation(dirPath: Path, execPath: Path, nTasks: int):
    # Write MOOSE args list
    args = [
        'mpirun',
        '-n',
        str(nTasks),
        execPath,
        '-i',
        INPUT_FILE
    ]

    # Run simulation
    subprocess.run(
        args,
        shell=False,
        cwd=dirPath,
        check=False
    )

def collectOutputs(dirPath: Path) -> dict:
    # Match MOOSE output location
    outputPath = dirPath / OUTPUT_FILE

    # Read in and extract data
    outputData = pd.read_csv(outputPath, index_col="time")
    outputDict = outputData.iloc[-1,:].to_dict()    # For steady state simulation we want last time

    return outputDict

def main():
    # Generate and populate directory for new simulation
    dirPath = directoryGenerator()

    # Update input file
    params = {
        'coolantTemp': 155,
        'protrusion': 0.001,
        'scale_therm_cond_W': 1.1,
        'scale_youngs_CuCrZr': 1.1
    }
    
    updateInputFile(dirPath, params)

    # Run simulation
    execPathStr = "../../proteus/proteus-opt"
    execPath = Path(execPathStr)
    nTasks = 16

    runSimulation(
        dirPath=dirPath,
        execPath=execPath,
        nTasks=nTasks
    )

    # Collect results
    results = collectOutputs(dirPath=dirPath)
    print(results)
    
    # Delete new simulation directory
    shutil.rmtree(dirPath)


if __name__ == '__main__':
    main()