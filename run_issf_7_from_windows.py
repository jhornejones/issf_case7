import subprocess

linuxSimulationPath = "/home/jhorne/projects/issf_case7"

command = f"wsl -e bash -lc 'cd {linuxSimulationPath} && python run_issf_7.py'"

subprocess.run(command)