from pathlib import Path
import uuid
import shutil

def main():
    dirPath = directoryGenerator()

def directoryGenerator() -> Path:
    # Create randomly named folder for current simulation
    dirName = str(uuid.uuid4())
    Path(dirName).mkdir(parents=True, exist_ok=True)
    dirPath = Path(dirName)

    # Copy MOOSE input file and data folder
    dataDir = "data"
    inputFile = "monoblock.i"

    dataDstPath = dirPath / dataDir
    shutil.copytree("data", dataDstPath, dirs_exist_ok=True)
    inputDstPath = dirPath / inputFile
    shutil.copyfile(inputFile, inputDstPath)

    return dirPath

if __name__ == '__main__':
    main()