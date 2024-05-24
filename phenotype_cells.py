import numpy as np
import pandas as pd
import anndata

def load_marker_dict_from_csv(file_path):
    """
    Load marker dictionary from a CSV file.
    """
    df = pd.read_csv(file_path, index_col=0)
    marker_dict = {}
    for marker in df.index:
        cell_types = df.loc[marker].dropna().tolist()
        marker_dict[marker] = cell_types
    return marker_dict

def phenotype_cells(adata, marker_dict, gate=0.5, label="phenotype", pheno_threshold_percent=None, pheno_threshold_abs=None, verbose=True):

    # Create a dataframe from the adata object
    data = pd.DataFrame(adata.layers['scaled'], columns=adata.var.index, index=adata.obs.index)

    def phenotype_cells(data, marker_dict, gate):
        # Initialize an empty DataFrame to store scores
        cell_types = list(set([ct for cell_types in marker_dict.values() for ct in cell_types]))
        scores = pd.DataFrame(0, index=data.index, columns=cell_types)
        marker_weights = {marker: 1.0 / len(cell_types) for marker, cell_types in marker_dict.items()}

        # Score cells based on marker expression
        for marker, cell_types in marker_dict.items():
            if marker in data.columns:
                expression = data[marker].values
                for cell_type in cell_types:
                    scores[cell_type] += (expression > gate).astype(int) * marker_weights[marker]

        # Assign cell types based on the highest score
        scores['max_score'] = scores.max(axis=1)
        scores['phenotype'] = scores.idxmax(axis=1)
        scores.loc[scores['max_score'] == 0, 'phenotype'] = 'Unknown'

        return scores['phenotype']

    # Annotate cells
    adata.obs[label] = phenotype_cells(data, marker_dict, gate)

    # Apply the phenotype threshold if given
    if pheno_threshold_percent or pheno_threshold_abs is not None:
        p = pd.DataFrame(adata.obs[label])

        # Function to remove phenotypes that are less than the given threshold
        def remove_phenotype(p, pheno_threshold_percent, pheno_threshold_abs):
            x = pd.DataFrame(p.groupby([label]).size())
            x.columns = ['val']
            # Find the phenotypes that are less than the given threshold
            if pheno_threshold_percent is not None:
                fail = list(x.loc[x['val'] < x['val'].sum() * pheno_threshold_percent / 100].index)
            if pheno_threshold_abs is not None:
                fail = list(x.loc[x['val'] < pheno_threshold_abs].index)
            p[label] = p[label].replace(dict(zip(fail, np.repeat('Unknown', len(fail)))))
            # Return
            return p

        # Apply the threshold removal function
        adata.obs[label] = remove_phenotype(p, pheno_threshold_percent, pheno_threshold_abs)[label]

    return adata

# Main function to be used when running the script directly
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='The phenotyping function takes in the `scaled data` and a marker dictionary to assign phenotype annotation to each cell in the dataset.')
    parser.add_argument('--adata', type=str, help='Path to the AnnData object (.h5ad file)')
    parser.add_argument('--marker_csv', type=str, help='Path to the marker dictionary CSV file')
    parser.add_argument('--gate', type=float, default=0.5, help='Threshold value for determining positive cell classification.')
    parser.add_argument('--label', type=str, default='phenotype', help='Name of the column where the phenotype annotations will be stored.')
    parser.add_argument('--pheno_threshold_percent', type=float, default=None, help='Minimum percentage of cells that must exhibit a particular phenotype to be considered valid.')
    parser.add_argument('--pheno_threshold_abs', type=int, default=None, help='Absolute number of cells that must exhibit a particular phenotype to be considered valid.')
    parser.add_argument('--verbose', type=bool, default=True, help='Print detailed progress messages.')
    args = parser.parse_args()

    # Load the AnnData object
    adata = anndata.read_h5ad(args.adata)

    # Load the marker dictionary from CSV
    marker_dict = load_marker_dict_from_csv(args.marker_csv)

    # Run phenotyping
    adata = phenotype_cells(adata, marker_dict=marker_dict, gate=args.gate, label=args.label, pheno_threshold_percent=args.pheno_threshold_percent, pheno_threshold_abs=args.pheno_threshold_abs, verbose=args.verbose)

    # Save the annotated data
    adata.write(args.adata)
