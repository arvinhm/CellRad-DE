from setuptools import setup, find_packages

setup(
    name='CellRad-DE',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'Pillow',
        'tifffile',
        'numpy',
        'matplotlib',
        'deepcell',
        'ome-types',
        'napari',
        'opencv-python-headless',
        'pandas',
        'scikit-image',
        'anndata',
        'scanpy',
        'scimap',
        'phenotype-cells'
    ],
)
