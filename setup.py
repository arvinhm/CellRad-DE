import subprocess
import sys
from setuptools import setup, find_packages, Command

class CreateCondaEnv(Command):
    description = "Create a Conda environment and install required packages"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        env_name = "CellRad-DE"  # Define the environment name here
        python_version = "3.10"
        requirements = [
            "Pillow",
            "tifffile",
            "numpy",
            "matplotlib",
            "deepcell",
            "ome-types",
            "napari",
            "opencv-python-headless",
            "pandas",
            "scikit-image",
            "anndata",
            "jupyter",
            "ipykernel",
            "imagecodecs",
            "scanpy",
            "scimap"
            # phenotype-cells is your custom package, handle it separately if needed
        ]

        # Create the Conda environment
        try:
            subprocess.run(["conda", "create", "-n", env_name, f"python={python_version}", "-y"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error creating Conda environment: {e}")
            sys.exit(1)

        # Activate the environment and install the packages
        failed_packages = []
        for package in requirements:
            try:
                subprocess.run(f"conda run -n {env_name} pip install {package}", shell=True, check=True)
            except subprocess.CalledProcessError:
                failed_packages.append(package)
                print(f"Failed to install {package}")

        if failed_packages:
            print(f"The following packages could not be installed: {', '.join(failed_packages)}")
        else:
            print(f"All packages installed successfully in the environment '{env_name}'.")

        # Install custom package
        try:
            subprocess.run(f"conda run -n {env_name} pip install -e .", shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing custom package: {e}")
            sys.exit(1)

with open('requirements.txt', 'w') as f:
    f.write("\n".join([
        "Pillow",
        "tifffile",
        "numpy",
        "matplotlib",
        "deepcell",
        "ome-types",
        "napari",
        "opencv-python-headless",
        "pandas",
        "scikit-image",
        "anndata",
        "jupyter",
        "ipykernel",
        "imagecodecs",
        "scanpy",
        "scimap"
    ]))

setup(
    name='CellRad-DE',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'setuptools',
    ],
    cmdclass={
        'create_conda_env': CreateCondaEnv,
    },
)
