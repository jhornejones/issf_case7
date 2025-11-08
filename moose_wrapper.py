from pathlib import Path
import uuid
import shutil
import subprocess
import pandas as pd
import copy

MOOSE_COMMENT = '#'
PARAM_START = "*_*START*_*"
PARAM_END = "*_*END*_*"

class MooseSim:
    _inputFile = None
    _multiAppInputFiles = None
    _outputFile = None
    _execPath = None
    _simPath = None
    
    def __init__(
            self,
            inputFile: str,
            outputFile: str,
            execPathStr: str,
            multiAppInputFiles: list = None,
            otherData: list = None
        ):
        
        if not Path(inputFile).is_file():
            raise FileExistsError("Input file does not exist")
        
        if not Path(execPathStr).is_file():
            raise FileExistsError("MOOSE executable does not exist")
        
        if not multiAppInputFiles is None:
            for inFile in multiAppInputFiles:
                if not Path(inFile).is_file():
                    raise FileExistsError("Multiapp input files do not exist")
        
        if not otherData is None:
            for data in otherData:
                if not Path(data).exists():
                    raise FileExistsError("Other data does not exist")
        
        self._inputFile = inputFile
        self._multiAppInputFiles = copy.deepcopy(multiAppInputFiles)
        self._outputFile = outputFile
        self._execPath = Path(execPathStr)
        self._simPath = self._directoryGenerator(otherData)
        
        print("MOOSE Wrapper: created MOOSE setup", flush=True)
    
    def __del__(self):
        shutil.rmtree(self._simPath)
        print("MOOSE Wrapper: deleted simulation directory")
            
    def _directoryGenerator(self, otherData) -> Path:
        # Create randomly named folder for current simulation
        dirName = str(uuid.uuid4())
        Path(dirName).mkdir(parents=True, exist_ok=True)
        dirPath = Path(dirName)

        # Copy MOOSE input file into new directory
        inputDstPath = dirPath / self._inputFile
        shutil.copyfile(self._inputFile, inputDstPath)
        
        # Copy MOOSE multiapp input files
        if not self._multiAppInputFiles is None:
            for inFile in self._multiAppInputFiles:
                inputDstPath = dirPath / inFile
                shutil.copyfile(inFile, inputDstPath)
        
        # If other data to copy, copy that as well
        if not otherData is None:
            for data in otherData:
                dataDstPath = dirPath / data
                shutil.copytree(data, dataDstPath, dirs_exist_ok=True)

        return dirPath

    # Content derived from mooseherder
    def updateInputFile(self, params: dict):
        # Update parent input files
        inputPath = self._simPath / self._inputFile
        self._updateAMOOSEInputFile(inputPath, params)
        
        # Update multiapp input files
        if not self._multiAppInputFiles is None:
            for inFile in self._multiAppInputFiles:
                inputPath = self._simPath / inFile
                self._updateAMOOSEInputFile(inputPath, params)

        print("MOOSE Wrapper: updated MOOSE input files", flush=True)        

    def _updateAMOOSEInputFile(self, inputPath: Path, params: dict):
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
                        variable, value, comment = self._splitMOOSEInputLine(line)
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
    def _splitMOOSEInputLine(self, strippedLine: str) -> tuple:
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

    # Run using subprocess
    def runSimulation(self, nTasks: int):
        
        print("MOOSE Wrapper: calling MOOSE", flush=True)

        # Write MOOSE args list
        args = [
            'mpirun',
            '-n',
            str(nTasks),
            self._execPath,
            '-i',
            self._inputFile,
            '--allow-unused'
        ]

        # Run simulation
        subprocess.run(
            args,
            shell=False,
            cwd=self._simPath,
            check=False,
        )

        print("MOOSE Wrapper: MOOSE call complete", flush=True)

    def collectSteadyStateOutputs(self) -> dict:
        # Match MOOSE output location
        outputPath = self._simPath / self._outputFile
        
        if not outputPath.is_file():
            raise FileExistsError("Output file does not exist")

        # Read in and extract data
        outputData = pd.read_csv(outputPath, index_col="time")
        outputDict = outputData.iloc[-1,:].to_dict()    # For steady state simulation we want last time

        return outputDict