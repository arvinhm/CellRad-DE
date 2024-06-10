from PIL import Image
import tifffile as tf
import numpy as np
import os
from matplotlib import pyplot as plt
from deepcell.applications import Mesmer
from deepcell.utils.plot_utils import create_rgb_image, make_outline_overlay
from ome_types import from_tiff
import napari
import xml.etree.ElementTree as ET
import cv2
from pathlib import Path
import pandas as pd
import skimage.io
from skimage.measure import regionprops_table
import anndata
import scanpy as sc
import scimap as sm
import json
from phenotype_cells import phenotype_cells, load_marker_dict_from_csv
import shutil
import subprocess
import tempfile

def subtract_background(input_path, output_path, fiji_path, radius=30):
    # Ensure the paths are absolute
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    fiji_path = os.path.abspath(fiji_path)
    
    # Check if the input file exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found at {input_path}. Please ensure the path is correct.")
    
    # Check if the Fiji executable exists
    if not os.path.exists(fiji_path):
        raise FileNotFoundError(f"Fiji executable not found at {fiji_path}. Please ensure the path is correct.")
    
    # Create the macro content using string concatenation
    macro_content = (
        'open("' + input_path.replace('\\', '\\\\') + '");\n' +
        'run("Subtract Background...", "rolling=' + str(radius) + ' stack");\n' +
        'saveAs("Tiff", "' + output_path.replace('\\', '\\\\') + '");\n' +
        'close();\n'
    )
    
    # Create a temporary macro file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.ijm') as temp_macro:
        temp_macro.write(macro_content.encode())
        temp_macro_path = temp_macro.name
    
    print(f"Temporary macro file created at {temp_macro_path}")
    
    # Run ImageJ with the macro
    result = subprocess.run([fiji_path, "--headless", "--run", temp_macro_path], capture_output=True, text=True)
    
    # Print the output from the subprocess
    print(result.stdout)
    print(result.stderr)
    
    # Clean up the temporary macro file
    os.remove(temp_macro_path)
    print(f"Temporary macro file {temp_macro_path} deleted")

    # Check if the output file was created
    if os.path.exists(output_path):
        print(f"Output file created successfully at {output_path}")
    else:
        print(f"Output file was not created at {output_path}")

#input_path = r'c:\Users\Yue\Desktop\FINAL_MC\BEMS340264_Scene-002_half.ome.tif'
#output_path = r'c:\Users\Yue\Desktop\FINAL_MC\BEMS340264_Scene-002_subtracted.ome.tif'
#fiji_path = r'c:\Users\Yue\Desktop\Fiji.app\ImageJ-win64.exe'  # Adjust this path to your Fiji installation

# Call the function
#subtract_background(input_path, output_path, fiji_path, radius=30)

def resize(input_path, output_path, resize_ratio):
    with tf.TiffFile(input_path) as tif:
        image_data = tif.asarray()  
        metadata = tif.ome_metadata  

        if image_data.ndim > 2:
            resized_channels = []
            for channel in image_data:
                img = Image.fromarray(channel)
                new_size = (img.width // resize_ratio, img.height // resize_ratio)
                resized_channels.append(np.array(img.resize(new_size)))
            resized_image_data = np.stack(resized_channels)
        else:
            img = Image.fromarray(image_data)
            new_size = (img.width // resize_ratio, img.height // resize_ratio)
            resized_image_data = np.array(img.resize(new_size))
    tf.imwrite(output_path, resized_image_data, metadata={'ome': metadata})


#input_path = 'TNPCRC_14/TNPCRC_14.ome.tif'
#output_path = 'TNPCRC_14/TNPCRC_14_half.ome.tif'
#resize_ratio = 2

# Call the function
#resize(input_path, output_path, resize_ratio)


def extract_pixel_size(file_path):
    """Extracts the pixel size from an OME-TIFF file."""
    ome_data = from_tiff(file_path)
    pixels = ome_data.images[0].pixels
    size_x = pixels.physical_size_x  # In microns
    size_y = pixels.physical_size_y  # In microns
    size_z = pixels.physical_size_z or 1  # In microns, default Z size is 1 if not provided
    resolution_x = pixels.size_x
    resolution_y = pixels.size_y
    if not size_x or not size_y:
        raise ValueError("No pixel size detected in metadata. Please enter manually.")
    return size_x, size_y, size_z, resolution_x, resolution_y

def segment_cells(input_path, output_path, dapi_idx, membrane_idx, pixel_size, token):
    os.environ['DEEPCELL_ACCESS_TOKEN'] = token
    def read_ome_tiff(file_path):
        with tf.TiffFile(file_path) as tif:
            image_data = tif.asarray()
            metadata = tif.ome_metadata
        return image_data, metadata
    def save_ome_tiff(file_path, image_data, metadata):
        tf.imwrite(file_path, image_data, metadata={'ome': metadata})

    def select_channels(image_data, dapi_idx, membrane_idx):
        if dapi_idx >= image_data.shape[0] or membrane_idx >= image_data.shape[0]:
            raise ValueError("Channel indices are out of bounds for the given image data.")
        
        dapi_channel = image_data[dapi_idx, :, :]
        membrane_channel = image_data[membrane_idx, :, :]
        return np.stack([dapi_channel, membrane_channel], axis=-1)

    image_data, metadata = read_ome_tiff(input_path)


    if image_data.ndim == 4:
        image_data = np.squeeze(image_data, axis=0)  


    channels_combined = select_channels(image_data, dapi_idx, membrane_idx)


    del image_data

    rgb_images = create_rgb_image(channels_combined[np.newaxis, ...], channel_colors=['blue', 'green'])

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].imshow(channels_combined[..., 0], cmap='Greys_r')
    ax[1].imshow(channels_combined[..., 1], cmap='Greys_r')
    ax[2].imshow(rgb_images[0, ...])
    ax[0].set_title('Nuclear channel')
    ax[1].set_title('Membrane channel')
    ax[2].set_title('Overlay')
    for a in ax:
        a.axis('off')
    plt.show()

    app = Mesmer()

    print('Training Resolution:', app.model_mpp, 'microns per pixel')

    if pixel_size == 'auto':
        size_x, size_y, size_z, resolution_x, resolution_y = extract_pixel_size(input_path)
        pixel_size = size_x  

    segmentation_predictions = app.predict(channels_combined[np.newaxis, ...], image_mpp=pixel_size)


    segmentation_predictions = segmentation_predictions.astype(np.uint16)

    overlay_data = make_outline_overlay(rgb_data=rgb_images, predictions=segmentation_predictions)

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(rgb_images[0, ...])
    ax[1].imshow(overlay_data[0, ...])
    ax[0].set_title('Raw data')
    ax[1].set_title('Predictions')
    for a in ax:
        a.axis('off')
    plt.show()

    save_ome_tiff(output_path, segmentation_predictions, metadata)
    print(f'Segmentation mask saved to {output_path}')

#input_path = 'processing\\BEMS340264_Scene-002.ome.tif'
#output_path = 'processing\\BEMS340264_Scene-002_Cell_mask_file.ome.tif'
#dapi_idx = 25 
#membrane_idx = 24  
#pixel_size = 'auto'  # Set to 'auto' to automatically calculate pixel size or enter manually
#token = 'e8MkSMdE.L6p85e7ToAR6UShqyzkSJP1m9H6WeZ2r'

#segment_cells(input_path, output_path, dapi_idx, membrane_idx, pixel_size, token)

def visualization(image_path, mask_path):
    multiplex_image = tifffile.imread(image_path)
    cell_mask = tifffile.imread(mask_path)
    viewer = napari.Viewer()
    if multiplex_image.ndim == 3 and multiplex_image.shape[0] > 1:
        for i in range(multiplex_image.shape[0]):
            viewer.add_image(multiplex_image[i], name=f'Channel {i + 1}', contrast_limits=[multiplex_image.min(), multiplex_image.max()])
    else:
        viewer.add_image(multiplex_image, name='Multiplex Image', contrast_limits=[multiplex_image.min(), multiplex_image.max()])
    viewer.add_labels(cell_mask, name='Cell Mask')
    napari.run()

#image_path = 'processing\\BEMS340264_Scene-002.ome.tif'
#mask_path = 'processing\\BEMS340264_Scene-002_Cell_mask_file.ome.tif'
#visualization(image_path, mask_path)




def write_mhd(filename, dim_size, element_spacing, element_type, data_file):
    mhd_content = f"""NDims = 3
DimSize = {dim_size[1]} {dim_size[0]} {dim_size[2]}
ElementSpacing = {element_spacing[0]} {element_spacing[1]} {element_spacing[2]}
ElementNumberOfChannels = 1
ElementByteOrderMSB = False
ElementType = {element_type}
ElementDataFile = {data_file}
"""
    with open(filename, 'w') as file:
        file.write(mhd_content)

def extract_pixel_size(metadata):

    root = ET.fromstring(metadata)
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    try:
        pixels = root.find('.//ome:Pixels', namespace)
        size_x = float(pixels.get('PhysicalSizeX')) / 1000
        size_y = float(pixels.get('PhysicalSizeY')) / 1000
        size_z = float(pixels.get('PhysicalSizeZ', 1)) / 1000  
        return size_x, size_y, size_z
    except (AttributeError, TypeError):
        raise ValueError("Pixel size information is missing in the OME-TIFF file metadata.")

def create_3d_model(image_path, mask_path, output_dir, channels, use_mask, pixel_size='auto', activity=1):

    with tifffile.TiffFile(image_path) as tif:
        images = tif.asarray()
        metadata = tif.ome_metadata


    if use_mask:
        with tifffile.TiffFile(mask_path) as tif:
            mask = tif.asarray().astype(bool)

    images = images.astype(np.float32)

    if pixel_size == 'auto':
        try:
            element_spacing = extract_pixel_size(metadata)
        except ValueError as e:
            print(e)
            raise
    else:
        element_spacing = (pixel_size, pixel_size, pixel_size)

    for channel in channels:
        channel_dir = os.path.join(output_dir, f'Channel_{channel}')
        os.makedirs(channel_dir, exist_ok=True)

        channel_image = images[channel]

        if use_mask:
            masked_image = np.where(mask, channel_image, 0)
        else:
            masked_image = channel_image

        dim_size = (masked_image.shape[0], masked_image.shape[1], 1)

        total_A = np.sum(masked_image)
        if total_A == 0:
            raise ValueError(f"Total activity in the masked area for channel {channel} is zero.")
        source_normalized = masked_image / total_A

        normalized_raw_file_path = os.path.join(channel_dir, 'Source_normalized.raw')
        with open(normalized_raw_file_path, 'wb') as file:
            file.write(source_normalized.astype(np.float32).tobytes())

        normalized_mhd_file_path = os.path.join(channel_dir, 'Source_normalized.mhd')
        write_mhd(normalized_mhd_file_path, dim_size, element_spacing, 'MET_FLOAT', os.path.basename(normalized_raw_file_path))

        total_a_file_path = os.path.join(channel_dir, 'Total_A_Bq.txt')
        with open(total_a_file_path, 'w') as file:
            file.write(f'{(total_A / activity):.2f}')

#image_path = 'processing\\BEMS340264_Scene-002.ome.tif'
#mask_path = 'processing\\masks\\binary_mask_Cancer cell.tif'  # path to the binary mask file
#output_dir = 'processing\\output_directory'
#channels = [16]  # List of channels to process [8, 17, 19, 24]
#use_mask = False  # whether to use the mask file or not, actualy ECM markers do not need mask
#pixel_size = 'auto'  # set to 'auto' or enter manually
#activity = 1  # set activity per pixel value

#create_3d_model(image_path, mask_path, output_dir, channels, use_mask, pixel_size, activity)


#######################################################################################  
############## TISSUE CONTOURING AND NUCLEI EXTRACTION ############## 
####################################################################################### 


def tissue_mask(input_file, output_path, channel, threshold_value=5000):
    """
    Create a mask around tissue in a specified channel of an OME-TIFF file.
    
    Parameters:
    input_file (str): Path to the input OME-TIFF file.
    output_path (str): Path to save the output mask file.
    channel (int): The channel index to use for contouring.
    threshold_value (int): The threshold value for binary conversion (default: 5000).
    
    Returns:
    mask (numpy.ndarray): The generated mask with contours around the tissue.
    """
    image = tiff.imread(input_file)

    if image is None:
        raise ValueError("Image could not be loaded, check the file path and format")
    
    if channel >= image.shape[0]:
        raise ValueError("Channel index is out of bounds for the given image data.")
    
    image_channel = image[channel, :, :]

    _, thresh = cv2.threshold(image_channel, threshold_value, 65535, cv2.THRESH_BINARY)


    thresh = np.uint8(thresh / 255)


    kernel = np.ones((25, 25), np.uint8)
    dilated = cv2.dilate(thresh, kernel, iterations=2)


    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    mask = np.zeros_like(image_channel, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    tiff.imwrite(output_path, mask)

    return mask

def nuclei_mask(input_file, output_path, channel, threshold_value=5000):
    """
    Create a mask for nuclei in a specified channel of an OME-TIFF file.
    
    Parameters:
    input_file (str): Path to the input OME-TIFF file.
    output_path (str): Path to save the output nuclei mask file.
    channel (int): The channel index to use for extracting nuclei.
    threshold_value (int): The threshold value for binary conversion (default: 5000).
    
    Returns:
    mask (numpy.ndarray): The generated mask for the nuclei.
    """
    image = tiff.imread(input_file)

    if image is None:
        raise ValueError("Image could not be loaded, check the file path and format")
    
    if channel >= image.shape[0]:
        raise ValueError("Channel index is out of bounds for the given image data.")
    
    image_channel = image[channel, :, :]

    _, thresh = cv2.threshold(image_channel, threshold_value, 65535, cv2.THRESH_BINARY)

    thresh = np.uint8(thresh / 255)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(image_channel, dtype=np.uint8)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)

    tiff.imwrite(output_path, mask)

    return mask


#input_file = 'processing\\BEMS340264_Scene-002.ome.tif'
#tissue_output_path = 'processing\\tissue_mask.tif'
#nuclei_output_path = 'processing\\nuclei_mask.tif'
#channel = 25  # Example channel index


#tissue_mask = tissue_mask(input_file, tissue_output_path, channel)

#nuclei_mask = nuclei_mask(input_file, nuclei_output_path, channel)



#######################################################################################  
############## CREATE A 3D MODEL FOR CT IMAGE BASED ON WHOLE TISSUE MASK ############## 
####################################################################################### 


def write_mhd(filename, dim_size, element_spacing, element_type, data_file):
    """Writes an MHD file with the specified parameters."""
    mhd_content = f"""NDims = 3
DimSize = {dim_size[0]} {dim_size[1]} {dim_size[2]}
ElementSpacing = {element_spacing[0]} {element_spacing[1]} {element_spacing[2]}
ElementNumberOfChannels = 1
ElementByteOrderMSB = False
ElementType = {element_type}
ElementDataFile = {data_file}
"""
    with open(filename, 'w') as file:
        file.write(mhd_content)

def extract_pixel_size(metadata):
    """Extracts the pixel size from OME metadata."""
    root = ET.fromstring(metadata)
    namespace = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    try:
        pixels = root.find('.//ome:Pixels', namespace)
        size_x = float(pixels.get('PhysicalSizeX')) / 1000
        size_y = float(pixels.get('PhysicalSizeY')) / 1000
        size_z = float(pixels.get('PhysicalSizeZ', 1)) / 1000  # default Z size is 1 if not provided
        return size_x, size_y, size_z
    except (AttributeError, TypeError):
        raise ValueError("Pixel size information is missing in the OME-TIFF file metadata.")

def ct_scan(mask_path, ome_tiff_path, output_dir):
    """Processes the binary mask and raw OME-TIFF file to create a CT image and MHD file."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    with tifffile.TiffFile(mask_path) as tif:
        mask = tif.asarray().astype(bool)

    with tifffile.TiffFile(ome_tiff_path) as tif:
        ome_metadata = tif.ome_metadata
        pixel_size = extract_pixel_size(ome_metadata)


    CT_image = np.full(mask.shape, -1050, dtype=np.int16)

    CT_image[mask] = 19 # soft tissue is 19

    raw_file_path = Path(output_dir) / 'CT.raw'
    with open(raw_file_path, 'wb') as file:
        file.write(CT_image.tobytes())

    dim_size = (CT_image.shape[1], CT_image.shape[0], 1)
    element_spacing = pixel_size 

    mhd_file_path = Path(output_dir) / 'CT.mhd'
    write_mhd(mhd_file_path, dim_size, element_spacing, 'MET_SHORT', raw_file_path.name)

    print(f"CT image and MHD file saved to {output_dir}")


#mask_path = 'processing\\tissue_mask.tif'
#ome_tiff_path = 'processing\\BEMS340264_Scene-002.ome.tif'
#output_dir = 'processing\\CT'
#ct_scan(mask_path, ome_tiff_path, output_dir)

# extract_single_cell_data

def extract_single_cell_data(mask_path, image_path, channel_names_path, output_dir):
    """Extract single cell data including intensities and spatial coordinates."""
    channel_names_df = pd.read_csv(channel_names_path)
    channel_names_df.drop_duplicates(subset=['marker_name'], inplace=True)
    channel_names_df = channel_names_df[channel_names_df['marker_name'] != 'DAPI']
    channel_names = channel_names_df['marker_name'].tolist()

    image_data = tifffile.imread(image_path)
    if len(image_data.shape) != 3:
        raise ValueError("Expected image_data to be a 3D array with shape (channels, height, width)")

    mask_data = skimage.io.imread(mask_path, plugin='tifffile')
    if mask_data.shape != image_data.shape[1:]:
        raise ValueError("Mask and image dimensions must match.")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_df = pd.DataFrame()

    for i, channel in enumerate(channel_names):
        props = regionprops_table(mask_data, intensity_image=image_data[i],
                                  properties=('label', 'centroid', 'area', 'mean_intensity'))
        channel_df = pd.DataFrame(props)
        channel_df.rename(columns={'mean_intensity': channel}, inplace=True)

        if results_df.empty:
            results_df = channel_df
        else:
            results_df = pd.merge(results_df, channel_df, on='label', suffixes=('', '_drop'))
            results_df.drop([col for col in results_df.columns if '_drop' in col], axis=1, inplace=True)

    results_df['Y_centroid'] = results_df['centroid-0']
    results_df['X_centroid'] = results_df['centroid-1']
    results_df.drop(columns=['centroid-0', 'centroid-1'], inplace=True)

    output_csv = Path(output_dir) / 'cell_data.csv'
    results_df.to_csv(output_csv, index=False)

    spatial_data = results_df[['label', 'Y_centroid', 'X_centroid']].to_numpy()
    np.save(Path(output_dir) / 'cell_data_spatial.npy', spatial_data)

    marker_columns = [col for col in results_df.columns if col in channel_names]
    adata = anndata.AnnData(X=results_df[marker_columns].values,
                            obs=results_df[['Y_centroid', 'X_centroid', 'area']],
                            var=pd.DataFrame(index=marker_columns))
    adata.write(Path(output_dir) / 'cell_data.h5ad')

    print(f"Data saved to {output_csv}, {Path(output_dir) / 'cell_data_spatial.npy'}, and {Path(output_dir) / 'cell_data.h5ad'}")
    return results_df

#output_dir = "processing\\cell_data"
#masks = "processing\\BEMS340264_Scene-002_Cell_mask_file.ome.tif"
#image = "processing\\BEMS340264_Scene-002.ome.tif"
#channel_names = "processing\\markers.csv"
#extract_single_cell_data(masks, image, channel_names, output_dir)



def create_color_map(unique_phenotypes):
    """Generate a specific color for each phenotype using the Set1 colormap."""
    num_phenotypes = len(unique_phenotypes)
    color_map = plt.cm.get_cmap('Set1', num_phenotypes)  
    colors = [color_map(i) for i in range(num_phenotypes)]  
    return dict(zip(unique_phenotypes, (np.array(colors)[:, :3] * 255).astype(np.uint8)))

def celltype_prediction(adata_path, marker_csv_path, mask_path, output_path):
    """Predict cell types, generate figures, and save results to CSV files."""
    Path(output_path).mkdir(parents=True, exist_ok=True)
    
    adata = sc.read(adata_path)
    adata.raw = adata
    adata = sm.pp.log1p(adata)
    adata.layers["scaled"] = sc.pp.scale(adata, copy=True).X

    marker_dict = load_marker_dict_from_csv(marker_csv_path)

    adata = phenotype_cells(adata, marker_dict=marker_dict, gate=0.5, label='phenotype', pheno_threshold_percent=5, pheno_threshold_abs=10, verbose=True)

    markers_adata = adata.var_names.tolist()
    ax = sc.pl.heatmap(
        adata,
        markers_adata,
        groupby="phenotype",
        layer="scaled",
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        dendrogram=True,
        swap_axes=True,
        figsize=(15, 8),
        show=False,
    )
    fig = plt.gcf()
    heatmap_path = Path(output_path) / 'heatmap_high_res.png'
    fig.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close(fig)  # Close the figure to free memory

    if 'phenotype' in adata.obs.columns:
        output_df = adata.obs[['phenotype']]
        output_df.index.name = 'barcode'
        barcodes_phenotypes_path = Path(output_path) / 'barcodes_and_phenotypes.csv'
        output_df.to_csv(barcodes_phenotypes_path)
    else:
        raise ValueError("Error: 'phenotype' column not found in the data.")

    phenotypes = pd.read_csv(barcodes_phenotypes_path, index_col='barcode')
    phenotypes.index = phenotypes.index.map(str)
    phenotypes = phenotypes.drop('0', errors='ignore')

    mask_data = skimage.io.imread(mask_path, plugin='tifffile')

    unique_phenotypes = phenotypes['phenotype'].unique()
    phenotype_to_color = create_color_map(unique_phenotypes)

    colorized_mask = np.zeros((*mask_data.shape, 3), dtype=np.uint8)
    color_map = np.zeros((np.max(mask_data) + 1, 3), dtype=np.uint8)

    for barcode, phenotype in phenotypes['phenotype'].items():
        color_map[int(barcode)] = phenotype_to_color[phenotype]

    colorized_mask = color_map[mask_data]

    colorized_mask_path = Path(output_path) / 'colorized_mask.tif'
    tifffile.imsave(colorized_mask_path, colorized_mask)

    for phenotype in unique_phenotypes:
        binary_mask = np.isin(mask_data, phenotypes.index[phenotypes['phenotype'] == phenotype].astype(int))
        binary_mask = (binary_mask * 255).astype(np.uint8)
        binary_output_path = Path(output_path) / f'binary_mask_{phenotype}.tif'
        tifffile.imsave(binary_output_path, binary_mask)

    plt.figure(figsize=(10, 10))
    plt.imshow(colorized_mask)
    plt.axis('off')
    legend_handles = [Patch(facecolor=np.array(color) / 255, edgecolor='none', label=phenotype) for phenotype, color in phenotype_to_color.items()]
    plt.legend(handles=legend_handles, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.title('Colorized Cell Mask by Phenotype')
    plt.tight_layout()
    colorized_legend_path = Path(output_path) / 'colorized_mask_legend.png'
    plt.savefig(colorized_legend_path, dpi=300, bbox_inches='tight')
    plt.close()  

    print(f"Data saved to {output_path}")

#adata_path = 'processing\\cell_data\\cell_data.h5ad'
#marker_csv_path = 'processing\\marker_cell_types.csv'
#mask_path = 'processing\\BEMS340264_Scene-002_Cell_mask_file.ome.tif'
#output_path = 'processing\\masks'
#celltype_prediction(adata_path, marker_csv_path, mask_path, output_path)





def nuclei_celltype(tumor_mask_path, nuclei_mask_path, output_path):
    """
    Retain only the nuclei mask regions that are within the tumor mask.
    
    Parameters:
    tumor_mask_path (str): Path to the tumor segmentation mask file.
    nuclei_mask_path (str): Path to the nuclei mask file.
    output_path (str): Path to save the resulting mask file.
    
    Returns:
    combined_mask (numpy.ndarray): The resulting mask with nuclei regions within the tumor.
    """
    tumor_mask = tifffile.imread(tumor_mask_path).astype(bool)

    nuclei_mask = tifffile.imread(nuclei_mask_path).astype(bool)

    if tumor_mask.shape != nuclei_mask.shape:
        raise ValueError("The dimensions of the tumor mask and nuclei mask do not match.")

    combined_mask = np.zeros_like(nuclei_mask, dtype=np.uint8)
    combined_mask[nuclei_mask & tumor_mask] = 255

    tifffile.imwrite(output_path, combined_mask)

    return combined_mask

#tumor_mask_path = 'processing\\masks\\binary_mask_Cancer cell.tif'
#nuclei_mask_path = 'processing\\nuclei_mask.tif'
#output_path = 'processing\\tumor_nuclei.tif'

#combined_mask = nuclei_celltype(tumor_mask_path, nuclei_mask_path, output_path)




def extract_pixel_size(file_path):
    """Extracts the pixel size from an OME-TIFF file."""
    ome_data = from_tiff(file_path)
    pixels = ome_data.images[0].pixels
    size_x = pixels.physical_size_x  # In microns
    size_y = pixels.physical_size_y  # In microns
    size_z = pixels.physical_size_z or 1  # In microns, default Z size is 1 if not provided
    resolution_x = pixels.size_x
    resolution_y = pixels.size_y
    return size_x, size_y, size_z, resolution_x, resolution_y

def update_mac_file(file_path, output_dir, size_x, size_y, size_z, resolution_x, resolution_y):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if 'SetCutInRegion' in line and 'phantom' in line:
            new_line = line.split()
            new_line[-2] = f"{size_x / 1000:.6f}"
            new_lines.append(' '.join(new_line) + '\n')
        elif '/gate/actor/dose3D/setVoxelSize' in line:
            new_lines.append(f'/gate/actor/dose3D/setVoxelSize         {size_x / 1000:.6f} {size_y / 1000:.6f} {size_z / 1000:.6f} mm\n')
        elif '/gate/actor/dose3D/setResolution' in line:
            new_lines.append(f'/gate/actor/dose3D/setResolution         {resolution_x} {resolution_y} 1\n')
        elif '/gate/source/Cu67Source/setPosition' in line:
            pos_x = (resolution_x * size_x / 2) * -1
            pos_y = (resolution_y * size_y / 2) * -1
            pos_z = (size_z / 2) * -1
            new_lines.append(f'/gate/source/Cu67Source/setPosition {pos_x:.4f} {pos_y:.4f} {pos_z:.4f} um\n')
        else:
            new_lines.append(line)

    new_file_path = os.path.join(output_dir, os.path.basename(file_path))
    with open(new_file_path, 'w') as new_file:
        new_file.writelines(new_lines)

def process_mac_files(input_dir, output_dir, ome_tiff_path):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    size_x, size_y, size_z, resolution_x, resolution_y = extract_pixel_size(ome_tiff_path)

    for file_name in os.listdir(input_dir):
        if file_name.endswith('.mac'):
            file_path = os.path.join(input_dir, file_name)
            update_mac_file(file_path, output_dir, size_x, size_y, size_z, resolution_x, resolution_y)

#input_dir = 'mac_sources'
#output_dir = 'processing'
#ome_tiff_path = 'processing\\BEMS340264_Scene-002.ome.tif'
#process_mac_files(input_dir, output_dir, ome_tiff_path)



def organizer(sample_name, channel_name, final_step=False):
    sample_dir = os.path.join(sample_name, f"mc_simulation_{channel_name}")
    os.makedirs(sample_dir, exist_ok=True)
    
    subdirs = ["data", "input", "mac", "output"]
    for subdir in subdirs:
        os.makedirs(os.path.join(sample_dir, subdir), exist_ok=True)
    
    mc_reqs_dir = 'mc_reqs'
    if os.path.exists(mc_reqs_dir):
        for item in os.listdir(mc_reqs_dir):
            s = os.path.join(mc_reqs_dir, item)
            d = os.path.join(sample_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    
    ct_dir = os.path.join('processing', 'CT')
    data_dir = os.path.join(sample_dir, 'data')
    if os.path.exists(ct_dir):
        for item in os.listdir(ct_dir):
            s = os.path.join(ct_dir, item)
            d = os.path.join(data_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, dirs_exist_ok=True)
            else:
                shutil.copy2(s, d)
    
    processing_dir = 'processing'
    mac_dir = os.path.join(sample_dir, 'mac')
    if os.path.exists(processing_dir):
        for item in os.listdir(processing_dir):
            if item.endswith('.mac'):
                s = os.path.join(processing_dir, item)
                shutil.copy2(s, mac_dir)
    
    channel_dir = os.path.join('processing', 'output_directory', channel_name)
    input_dir = os.path.join(sample_dir, 'input')
    if os.path.exists(channel_dir):
        for item in os.listdir(channel_dir):
            s = os.path.join(channel_dir, item)
            d = os.path.join(input_dir, item)
            shutil.move(s, d)
    
    if final_step:
        remaining_files_dir = os.path.join(sample_name, 'files')
        os.makedirs(remaining_files_dir, exist_ok=True)
        for item in os.listdir(processing_dir):
            s = os.path.join(processing_dir, item)
            d = os.path.join(remaining_files_dir, item)
            if os.path.isdir(s):
                shutil.move(s, d)
            else:
                shutil.move(s, remaining_files_dir)

#sample_name = 'BEMS340264_Scene-002'
#channel_name = 'channel_24'  # User-defined channel name
#final_step = True  # Whether to move remaining files or not

#organizer(sample_name, channel_name, final_step)
