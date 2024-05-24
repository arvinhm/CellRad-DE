
# Cell-Level Radiation Dosimetry Explorer (CellRad-DE)

**Cell-Level Radiation Dosimetry Explorer (CellRad-DE)** is an advanced, scalable toolkit designed to analyze radiation dosimetry at the cellular level comprehensively. This tool uses high-dimensional multiplexed images as inputs to acquire high-resolution, multiplexed data. These data inputs enable precise cell segmentation and automatic cell annotation, which is critical for detailed cellular analysis. The software predicts the energy deposition of various radionuclides within different cell types based on specific markers of interest. Utilizing advanced algorithms, this toolkit can accurately model the spatial distribution of radionuclide energy deposition at the subcellular level. This capability is essential for understanding the differential effects of targeted radionuclide therapies at the cellular level. In addition to its robust analytical capabilities, the software is designed to be efficient and user-friendly, providing an intuitive interface that facilitates complex data analysis without requiring extensive computational expertise. This makes it an invaluable tool for molecular radiology and radiation oncology researchers, enabling them to conduct detailed dosimetry studies, optimize therapeutic strategies, and ultimately enhance the precision and efficacy of cancer treatments.

## Key Features

- **Multiplex Imaging Integration**: CellRad-DE supports all different imaging techniques, including Cyclic Immunofluorescence (CycIF), Multiplexed Ion Beam Imaging (MIBI), and Imaging Mass Cytometry (IMC), allowing for high-dimensional, multiplexed data analysis.
- **Automated Cell Segmentation and Annotation**: The software generates precise cell segmentation masks and automatically annotates cells based on provided markers, streamlining the analysis process.
- **Monte Carlo Simulation**: Leveraging sophisticated Monte Carlo simulations, CellRad-DE predicts dose uptake within each cell type, focusing on nuclei to assess DNA damage accurately.
- **Marker and Radionuclide Optimization**: The tool helps researchers determine the most effective radiation oncology methods and radionuclides for specific markers. It also predicts the optimal markers for each radionuclide, enhancing the precision of targeted therapies.
- **User-Friendly Interface**: Designed with the user in mind, CellRad-DE offers an intuitive interface that simplifies complex analyses, making it accessible to both novice and experienced researchers.

## Application and Benefits

CellRad-DE is a toolkit for advancing the understanding of cellular damage from radionuclide therapy. By integrating multiplex imaging data and performing detailed dosimetry analyses, this software enables researchers to:
- Identify the best radionuclides for use in radionuclide therapy based on different cellular markers.
- Predict the optimal radionuclide therapy doses to minimize the adverse effects.
- Assess potential DNA damage across various cell types, aiding in developing more effective and personalized radionuclide therapies.

## Contributors
Arvin Haj-Mirzaian; Victor Valladolid Onecha; Alejandro Bertolet Reina; Pedram Heidari

## Step-by-Step Tutorial
**Install**
- Step 1: Install Conda or Miniconda
  - Install Miniconda
    Download the Miniconda installer for your operating system from the Miniconda download page (https://docs.anaconda.com/free/miniconda/index.html).
    Run the installer and follow the installation instructions.

- Step 2: Install Git
  - Linux:
```bash
sudo apt-get update
sudo apt-get install git -y
```
- macOS:
```bash
brew install git
```
- Windows:
Download the Git for Windows installer from the Git for Windows download page (https://git-scm.com/downloads).
Run the installer and follow the installation instructions.

- Step 3: Clone the Repository and installation
  The installation autmatically create a conda enviroment named 'CellRad-DE'. Open a terminal (or Git Bash on Windows) and run the following command:
```bash
git clone https://github.com/arvinhm/CellRad-DE.git
cd CellRad-DE
python setup.py create_conda_env
conda activate CellRad-DE
```


## License
<p xmlns:cc="http://creativecommons.org/ns#" xmlns:dct="http://purl.org/dc/terms/"><a property="dct:title" rel="cc:attributionURL" href="https://github.com/arvinhm/CellRad-DE">Cell-Level Radiation Dosimetry Explorer (CellRad-DE)</a> by <a rel="cc:attributionURL dct:creator" property="cc:attributionName" href="https://github.com/arvinhm">Arvin Haj-Mirzaian, Victor Valladolid Onecha, Alejandro Bertolet Reina, and Pedram Heidari</a> is licensed under <a href="https://creativecommons.org/licenses/by-nc-nd/4.0/?ref=chooser-v1" target="_blank" rel="license noopener noreferrer" style="display:inline-block;">CC BY-NC-ND 4.0<img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/cc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/by.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nc.svg?ref=chooser-v1" alt=""><img style="height:22px!important;margin-left:3px;vertical-align:text-bottom;" src="https://mirrors.creativecommons.org/presskit/icons/nd.svg?ref=chooser-v1" alt=""></a></p>
