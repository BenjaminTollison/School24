import subprocess
import sys
import os
import shutil
import stat
import argparse


def CreateVenv(venv_dir="heli_env"):
    """
    Creates a virtual environment and provides instructions to activate it.

    Args:
        venv_dir (str): Directory where the virtual environment will be created.
    """
    # Step 1: Create the virtual environment
    try:
        subprocess.check_call([sys.executable, "-m", "venv", venv_dir])
        print(f"Virtual environment created successfully in '{venv_dir}'!")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        sys.exit(1)

    # Step 2: Set execute permissions for the activate script
    activate_script = (
        os.path.join(venv_dir, "bin", "activate")
        if os.name != "nt"
        else os.path.join(venv_dir, "Scripts", "activate.bat")
    )
    if os.path.exists(activate_script):
        os.chmod(activate_script, os.stat(activate_script).st_mode | stat.S_IXUSR)
        print(f"Granted execute permissions to '{activate_script}'")

    # Step 3: Provide activation instructions
    if os.name == "nt":  # Windows
        activation_command = f"{venv_dir}\\Scripts\\activate"
    else:  # macOS/Linux
        activation_command = f"source {venv_dir}/bin/activate"

    print(
        f"\nTo activate the virtual environment, run the following command in your terminal:\n"
    )
    print(f"    {activation_command}\n")
    # Prompt the user to confirm activation
    print(
        "After activating the virtual environment, rerun this script with the 'python3 GradingPurposes.py --gpu-optimization' option to proceed OR '--proceed' to continue on CPU"
    )
    sys.exit(0)  # Exit after providing instructions


def InstallCuPy():
    # Step 3: Prompt the user to check CUDA version
    print("\nBefore continuing, let's check your CUDA version.")
    print("Choose a method to check your CUDA version:")
    print("1. Use 'nvcc --version' (requires CUDA toolkit to be installed).")
    print("2. Use 'nvidia-smi' (requires NVIDIA driver to be installed).")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "1":
        # Method 1: Use `nvcc`
        try:
            result = subprocess.check_output(
                ["nvcc", "--version"], stderr=subprocess.STDOUT
            )
            print("\nCUDA version (Method 1):")
            print(result.decode())
            method = "pip install cupy-cuda11x (e.g., `cuda11.5`)"
        except FileNotFoundError:
            print(
                "Error: `nvcc` command not found. Make sure the CUDA toolkit is installed."
            )
            choice = input("Enter 2: ").strip()
        except subprocess.CalledProcessError as e:
            print(f"Error while checking CUDA version using `nvcc`: {e}")
            choice = input("Enter 2: ").strip()
    elif choice == "2":
        # Method 2: Use `nvidia-smi`
        try:
            result = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT)
            print("\nCUDA version (Method 2):")
            print(result.decode())
            method = "pip install cupy-cuda11x (e.g., `cuda11.5`)"
        except FileNotFoundError:
            print(
                "Error: `nvidia-smi` command not found. Make sure the NVIDIA driver is installed."
            )
        except subprocess.CalledProcessError as e:
            print(f"Error while checking CUDA version using `nvidia-smi`: {e}")
    else:
        print(
            "Invalid choice. Please rerun the script from the beginning and enter 1 or 2."
        )
        sys.exit(1)

    print(
        "to install run ",
        method,
        "rerun this script with the '--proceed' option to proceed.",
    )
    # Exit after providing instructions and checking CUDA version
    sys.exit(0)


def InstallRequiredPackages():
    requirements_file = "requirements.txt"

    # Check if the requirements file exists
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        sys.exit(1)

    # Run pip install -r requirements.txt
    try:
        subprocess.check_call(["python3", "-m", "ensurepip", "--upgrade"])
        subprocess.check_call(["pip", "install", "-r", requirements_file])
        print("All requirements installed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error occurred during installation: {e}")
        sys.exit(1)


def GenerateAllPlots():
    from Problem1 import SubPlotProblem1
    from Problem2 import (
        PlotProblem2,
        power_factor_no_tip_loss,
        power_factor_two_blades,
        power_factor_four_blades,
    )
    from Problem3 import PlotProblem3
    from Problem4 import PlotProblem4
    from Problem5 import PlotProblem5

    try:
        import cupy as cp

        GPU_FREEDOM = True
    except ModuleNotFoundError:
        GPU_FREEDOM = False

    SubPlotProblem1()
    PlotProblem2()
    print(r"$\kappa_{notiploss}$ =", power_factor_no_tip_loss)
    print(r"$\kappa_{n_b=2} =$", power_factor_two_blades)
    print(r"$\kappa_{n_b=4} =$", power_factor_four_blades)
    PlotProblem3()
    PlotProblem4()
    PlotProblem5()
    if GPU_FREEDOM:
        from Problem5 import FigureOfMerit3DPlot

        FigureOfMerit3DPlot()
    else:
        from Problem5backup import FigureOfMerit3DPlot

        print("CPU Bound")
        FigureOfMerit3DPlot()


def DeleteTempVenv(venv_dir="heli_env"):
    """
    Deletes the specified virtual environment directory.

    Args:
        venv_dir (str): Directory where the virtual environment is located.
    """
    # Check if the directory exists
    if os.path.exists(venv_dir) and os.path.isdir(venv_dir):
        try:
            shutil.rmtree(venv_dir)  # Remove the directory and its contents
            print(f"Virtual environment '{venv_dir}' deleted successfully!")
        except Exception as e:
            print(f"Error deleting virtual environment: {e}")
    else:
        print(f"Virtual environment '{venv_dir}' does not exist or is not a directory.")


def main():
    parser = argparse.ArgumentParser(
        description="Run tasks with a virtual environment."
    )
    parser.add_argument(
        "--create-venv",
        action="store_true",
        help="Create the virtual environment and exit.",
    )
    parser.add_argument(
        "--gpu-optimization",
        action="store_true",
        help="Adds gpu optimization with CuPy.",
    )
    parser.add_argument(
        "--proceed",
        action="store_true",
        help="Continue with package installation and other tasks.",
    )
    args = parser.parse_args()

    if args.create_venv:
        CreateVenv()  # Create the virtual environment and exit
    elif args.gpu_optimization:
        InstallCuPy()  # Install Cupy to faster speeds
    elif args.proceed:
        InstallRequiredPackages()  # Install required packages
        GenerateAllPlots()  # Generate plots
        DeleteTempVenv()  # Delete the virtual environment
    else:
        print(
            "Usage: python GradingPurposes.py --create-venv OR python GradingPurposes.py --proceed OR python GradingPurposes.py --gpu-optimization"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
