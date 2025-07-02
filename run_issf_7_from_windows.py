import subprocess

linuxSimulationPath = "/home/jhorne/projects/issf_case7"

command = f"wsl -e bash -lc \"cd {linuxSimulationPath} && source .venv/bin/activate && python run_issf_7.py\""

subprocess.run(command)