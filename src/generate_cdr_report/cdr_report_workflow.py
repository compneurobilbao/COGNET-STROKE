from argparse import ArgumentParser

from tqdm import tqdm
import nibabel as nib
import numpy as np
from neuromaps import stats, parcellate
import brainsmash
from brainsmash.mapgen.eval import sampled_fit
from brainsmash.mapgen.sampled import Sampled
from brainsmash.workbench.geo import volume
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

import pandas as pd
from statsmodels.stats import multitest
import os
from nilearn import plotting
from src.cdr import compute_cdr, compute_overlap
from src.generate_cdr_report import data

def _setup_parser():
    parser = ArgumentParser(description="Generate CDR report based on Neurosynth concepts.")
    parser.add_argument(
        "--lesion_data_dir",
        type=str,
        required=True,
        help="Path to directory where the patient lesion masks are stored.",
    )

    parser.add_argument(
        "--lnm_data_dir",
        type=str,
        required=True,
        help="Path to directory where the lnm maps are stored.",
    )

    parser.add_argument(
        "--concept_data_dir",
        type=str,
        required=True,
        help="Path to directory where the concept maps are stored.",
    )

    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Path to output directory where the reports will be saved.",
    )

    parser.add_argument(
        "--n_null",
        type=int,
        default=5000,
        help="Number of null permutations."
    )

    parser.add_argument(
        "--brain_parcellation",
        type=str,
        default=data.PARCELLATION,
        help="Brain parcellation for reducing computational cost of null maps generation."
    )


    return parser


def _check_args(parser):
    args = parser.parse_args()

    # Check if lesion path exists
    if not os.path.exists(args.lesion_data_dir):
        raise FileNotFoundError(f"Lesion mask folder {args.lesion_data_dir} does not exist.")
    
    # Check if lnm maps path exists
    if not os.path.exists(args.lnm_data_dir):
        raise FileNotFoundError(f"LNM data folder {args.lnm_data_dir} does not exist.")
    
    # Check if concept maps path exists
    if not os.path.exists(args.concept_data_dir):
        raise FileNotFoundError(f"Concept data folder {args.concept_data_dir} does not exist. Please run the generate_concept_maps workflow first.")
    
    # Check if output path exists
    if not os.path.exists(args.out_dir):
        raise FileNotFoundError(f"Output folder {args.out_dir} does not exist.")

    # Check if design matrix file exists
    if args.n_null is None:
        raise FileNotFoundError("You must provide a number of null maps.")

    # Check if contrast file exists
    if not os.path.exists(args.brain_parcellation):
        raise FileNotFoundError(f"Parcellation image {args.brain_parcellation} does not exist.")

    return args


def main():
    parser = _setup_parser()
    args = _check_args(parser)

    # Load the parcellation image
    parcellation = nib.load(args.brain_parcellation)
    parcellation_volume = parcellation.get_fdata()

    # Get the ROI centroids to be used for null maps generation
    temp_dir = data.TEMP_DIR
    centroids = []

    # Iterate over each parcel ID to calculate the centroid
    for parcel_id in np.unique(parcellation_volume):
        if parcel_id == 0:
            continue
        # Get the coordinates of the voxels in the current parcel
        coords = np.argwhere(parcellation_volume == parcel_id)  # Get voxel indices
        
        # Calculate the centroid as the mean of the coordinates
        centroid = coords.mean(axis=0)  # mean across x, y, z dimensions
        centroids.append(np.round(centroid).astype(int))

    # Save the centroids to a text file
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    np.savetxt(os.path.join(temp_dir, 'parcellation_coordinates.txt'), centroids, fmt='%d')

    # Initialize the parcellater
    parcellater = parcellate.Parcellater(parcellation, space='MNI152', resampling_target='data')

    # Make the distance matrix, coordinates are calculated from the centroids of each ROI, this is used to create null maps later
    coord_file = os.path.join(temp_dir, 'parcellation_coordinates.txt')

    filenames = brainsmash.workbench.geo.volume(coord_file, temp_dir)

    # These are three of the key parameters affecting the variogram fit

    kwargs = {'ns': 250,
            'knn': 200,
            'pv': 70, 
            'nh':50,
            }

    # Load the concept maps and generate the data structure
    concept_names = ['motor', 'eye_movements', 'motion_perception', 'visual_perception', 'auditory_perception','tactile_perception', 'pain', 
                'action','facial_perception', 'multisensory_integration', 'attention',  'working_memory', 'inhibition',
                'memory', 'language', 'numerical_cognition', 'cognitive_control', 'imagery', 'social_cognition', 'emotion', 'decision_making',
                'reward_processing']

    concept_maps = {}

    # Set directory for maps
    maps_dir = os.path.join(args.concept_data_dir, 'corrected_fdr')

    for name in concept_names:
        # Load the map for the corresponding term ID
        concept_image_file = f"{name}_z_desc-association_level-voxel_corr-FDR_method-indep.nii.gz"
        concept_image = nib.load(os.path.join(maps_dir, concept_image_file))
        
        # Create mask for significant values  
        parcellated_concept = parcellater.fit_transform(concept_image, space='MNI152')
        parcellated_concept_mask = np.where(parcellater.fit_transform(concept_image, space='MNI152') > 2, 1, 0) 
        out_of_concept_rois = (np.ones_like(parcellated_concept_mask) - parcellated_concept_mask) == 1
        parcellated_concept_mask[out_of_concept_rois] = -1
        # Store the maps in the dictionary under the new name
        concept_maps[name] = {
            'parcellated_concept_mask_values': parcellated_concept_mask,
            'parcellated_concept_values': parcellated_concept,
        }
    # Load the patients data and generate the data structure    
    # Specify the folder paths

    # Display an example of how work the null maps generation parameters
    sampled_fit(concept_maps['working_memory']['parcellated_concept_values'].squeeze(), filenames['D'], filenames['index'], nsurr=10, **kwargs)
    # store the image generated by the sampled_fit function
    plt.savefig(os.path.join(temp_dir, 'sampled_fit_example.png'))
    plt.close()

    # Get a list of all files in the functional disconnectivity folder
    file_names = os.listdir(args.lnm_data_dir)

    # Filter for .nii.gz files
    nii_files = [f for f in file_names if f.endswith('.nii.gz')]
    sub_names = [f.replace('.nii.gz', '') for f in nii_files]

    # Make an empty dictionary to store subject data
    sub_maps = {}

    # Iterate through each file and process
    for sub in sub_names:
        lesion_file = f"{sub}.nii.gz"
        fdis_file = f"{sub}.nii.gz"
        
        try:
            # Load the functional disconnectivity image
            disconnectivity_path = os.path.join(args.lnm_data_dir, fdis_file)
            sub_image = nib.load(disconnectivity_path)
            sub_volume = sub_image.get_fdata()


            # Load the corresponding lesion image
            lesion_path = os.path.join(args.lesion_data_dir, lesion_file)
            if not os.path.exists(lesion_path):
                raise FileNotFoundError(f"Lesion file not found for {sub}")

            sub_lesion_image = nib.load(lesion_path)

            # Separate positive and negative values from the disconnectivity image
            positive_volume = np.where(sub_volume > 0, sub_volume, 0).astype(np.float32)
            positive_image = nib.Nifti1Image(positive_volume, sub_image.affine)
            
            negative_volume = np.where(sub_volume < 0, -sub_volume, 0).astype(np.float32)
            negative_image = nib.Nifti1Image(negative_volume, sub_image.affine)

            # Parcellate the lesion image
            parcellated_lesion = np.where(parcellater.fit_transform(sub_lesion_image, space='MNI152') > 0, 1, 0)

            # Parcellate the positive volume
            parcellated_possub = parcellater.fit_transform(positive_image, space='MNI152')


            # Parcellate the negative volume
            parcellated_negsub = parcellater.fit_transform(negative_image, space='MNI152')


            # Store the processed data for the subject in the sub_maps dictionary
            sub_maps[sub] = {
                'parcellated_lesion_values': parcellated_lesion,
                'parcellated_values_positive': parcellated_possub,
                'parcellated_values_negative': parcellated_negsub
            }

        except FileNotFoundError as e:
            print(f"{e}: Skipping {sub}")
        except Exception as e:
            print(f"Unexpected error processing {sub}: {e}")

    # Output summary
    print(f"Processed {len(sub_maps)} subjects successfully.")

    # Compute CDR for each concept map
    pos_cdr= []
    neg_cdr = []
    posp = []
    negp = []
    all_subs = []
    all_concepts = []
    lesion_overlaps_p= []
    lesion_overlaps = []

    for idx, concept in enumerate(concept_maps.keys()):
        tqdm.write(f"Processing concept {idx + 1}/{len(concept_maps.keys())}: {concept}")
        
        # Generate nulls for the main concept array
        nulls = Sampled(x=concept_maps[f'{concept}']['parcellated_concept_values'].squeeze(), D=filenames['D'], index=filenames['index'], n_jobs=-1, **kwargs)(args.n_null)
        nulls_thr = np.where(nulls > 2, 1, -1)

        for idx, sub in tqdm(enumerate(sub_maps.keys())):
            all_subs.append(sub)
            all_concepts.append(concept)
            
            # Calculate correlations for the neuroquery array
            cdr_lnm_positive, p_val_pos = compute_cdr(concept_maps[f'{concept}']['parcellated_concept_mask_values'].squeeze(), 
                                                    sub_maps[f'{sub}']['parcellated_values_positive'].squeeze(), 
                                                    nulls_thr)
            cdr_lnm_negative, p_val_neg = compute_cdr(concept_maps[f'{concept}']['parcellated_concept_mask_values'].squeeze(),
                                                    sub_maps[f'{sub}']['parcellated_values_negative'].squeeze(), 
                                                    nulls_thr)

            lesion_overlap, p_val_overlap = compute_overlap(concept_maps[f'{concept}']['parcellated_concept_mask_values'].squeeze(),
                                                        sub_maps[f'{sub}']['parcellated_lesion_values'].squeeze(),  
                                                        nulls_thr)
            # Append correlation results for the neuroquery array
            pos_cdr.append(cdr_lnm_positive)
            posp.append(p_val_pos)
            neg_cdr.append(cdr_lnm_negative)
            negp.append(p_val_neg)
            lesion_overlaps.append(lesion_overlap)
            lesion_overlaps_p.append(p_val_overlap)

    df_subs = pd.DataFrame({'sub': all_subs, 
                            'concept': all_concepts, 
                            'pos_cdr': pos_cdr, 
                            'posp': posp, 
                            'neg_cdr': neg_cdr, 
                            'negp': negp,
                            'lesion_overlap': lesion_overlaps,
                            'lesion_overlap_p': lesion_overlaps_p})
    
    # Correct p-values for multiple comparisons and dump the results file
    # Loop through each unique subject
    for x in df_subs['sub'].unique():
        df_sub = df_subs[df_subs['sub'] == x]  # Filter for the current subject
        
        # Apply multiple testing correction for 'pos_corr_p' column
        significance_lesion, corrected_lesion, _, _ = multitest.multipletests(df_sub['lesion_overlap_p'], alpha=0.05, method='bonferroni')
        
        # Apply multiple testing correction for 'pos_corr_p' column
        significance_pos, corrected_pos, _, _ = multitest.multipletests(df_sub['posp'], alpha=0.05, method='bonferroni')
        
        # Apply multiple testing correction for 'neg_corr_p' column
        significance_neg, corrected_neg, _, _ = multitest.multipletests(df_sub['negp'], alpha=0.05, method='bonferroni')
        
        # Add the results back to the dataframe
        df_subs.loc[df_subs['sub'] == x, 'significance_lesion'] = significance_lesion
        df_subs.loc[df_subs['sub'] == x, 'corrected_p_lesion'] = corrected_lesion

        df_subs.loc[df_subs['sub'] == x, 'significance_pos'] = significance_pos
        df_subs.loc[df_subs['sub'] == x, 'corrected_p_pos'] = corrected_pos
        
        df_subs.loc[df_subs['sub'] == x, 'significance_neg'] = significance_neg
        df_subs.loc[df_subs['sub'] == x, 'corrected_p_neg'] = corrected_neg

    df_subs["significance_pos"] = df_subs["significance_pos"].astype(bool)
    df_subs["significance_neg"] = df_subs["significance_neg"].astype(bool)

    # Save the results to a CSV file
    df_subs.to_csv(os.path.join(args.out_dir, 'sub_concept_to_disconnection_representation.csv'), index=False)
    
    # Genrate a general report counting the number of significant CDRs per subject
    df_subs_positive = df_subs[df_subs['corrected_p_pos'] < 0.05]
    df_subs_negative = df_subs[df_subs['corrected_p_neg'] < 0.05]

    positive_concepts = df_subs_positive['concept'].value_counts()
    negative_concepts = df_subs_negative['concept'].value_counts()

    # Plotting Positive Correlations
    plt.figure(figsize=(12, 6))
    sns.barplot(x=positive_concepts.index, y=positive_concepts.values, color='gray')
    plt.title('Concepts with significant positive Concept to Disconnection Representations (CDR)')
    plt.xlabel('Concepts')
    plt.ylabel('#CDRs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'positive_cdrs.svg'), format='svg', dpi=300)
    plt.close()

    # Plotting Negative Correlations
    plt.figure(figsize=(12, 6))
    sns.barplot(x=negative_concepts.index, y=negative_concepts.values, color='gray')
    plt.title('Concepts with significant negative Concept to Disconnection Representations (CDR)')
    plt.xlabel('Concepts')
    plt.ylabel('#CDRs')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, 'negative_cdrs.svg'), format='svg', dpi=300)
    plt.close()

    # Generate a report folder with a report file per subject

    # Create a directory to save the images if it doesn't exist
    output_folder = os.path.join(args.out_dir, 'patient_CDR')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Group by each subject
    for subject, subject_data in df_subs.groupby('sub'):
        # Concept associations
        subject_data_signif = subject_data.copy()
        subject_data_signif.loc[~(subject_data_signif['significance_pos']), 'pos_cdr'] = np.nan
        subject_data_signif.loc[~(subject_data_signif['significance_neg']), 'neg_cdr'] = np.nan

        concept_association_scores = subject_data_signif[['pos_cdr', 'neg_cdr']]
        
        # Add stars to significant association_scores
        normalized_scores = (concept_association_scores / concept_association_scores.max())
        normalized_scores = normalized_scores.set_index(subject_data_signif['concept'])
        normalized_scores = normalized_scores.rename(columns={'pos_cdr': 'pos_disconnectivity,\n max val(' + str(concept_association_scores.max()['pos_cdr'].round(1)) + ')', 
                                                            'neg_cdr': 'neg_disconnectivity,\n max val(' + str(concept_association_scores.max()['neg_cdr'].round(1)) + ')'})
        # plot heatmaps
        fig, ax  = plt.subplots(figsize=(6, 6))
        sns.heatmap(normalized_scores.reindex(index=concept_names), 
                    annot=True, 
                    fmt=".2f", 
                    cmap='coolwarm', 
                    center=0, 
                    linewidths=0.6, 
                    linecolor='black', 
                    ax=ax, 
                    cbar_kws={'shrink': 0.5},
                    vmin=0,
                    vmax=1
                    )
        ax.set_title(f"Concept to Disconnection Representation {subject}")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right", fontsize=10)
        plt.tight_layout()
        #adjust lines width
        for _, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
        plt.savefig(f"{output_folder}/{subject}_CDR.png", bbox_inches="tight")
        plt.close()
if __name__ == "__main__":
    main()