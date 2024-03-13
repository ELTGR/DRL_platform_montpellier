import subprocess
import time

tensorboard_command = f"x-terminal-emulator -e tensorboard --logdir="+"/home/ia/Desktop/generic_platform/Scenarios/UUV_Mono_Agent_TSP/models"
process_terminal_1 = subprocess.Popen(tensorboard_command, shell=True)