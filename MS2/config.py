
import subprocess
import sys




def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def setup():
    print("installing torch-fidelity...")
    install("torch-fidelity")
    print("done")