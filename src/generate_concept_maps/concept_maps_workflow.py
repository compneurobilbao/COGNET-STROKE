import nimare
import os
import json
from argparse import ArgumentParser
from src.generate_concept_maps import data
def _setup_parser():
    parser = ArgumentParser(description="Download and build concept maps based on neurosynth database")
    parser.add_argument(
        "--out_dir",
        required=True,
        type=str,
        help="Path to output directory where the concept maps will be saved.",
    )

    parser.add_argument(
        "--vocab",
        type=str,
        default="LDA50",
        help="Neurosynth topic vocabulary code (e.g., 'LDA50', 'LDA100', ...)."
    )

    parser.add_argument(
        "--concept_mapping",
        type=str,
        default=data.LDA_50,
        help="Path to the dictionary renaming the LDA topics to the concepts."
    )


    return parser


def _check_args(parser):
    args = parser.parse_args()

    # Check if input file exists
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)
    elif not os.path.isdir(args.out_dir):
        raise NotADirectoryError(f"The output directory {args.out_dir} is not a directory.")

    # Check if design matrix file exists
    if args.vocab is None:
        raise FileNotFoundError("You must provide a vocab code.")

    # Check if contrast file exists
    if not os.path.exists(args.concept_mapping):
        raise FileNotFoundError(f"Contrast file {args.concept_mapping} does not exist.")

    return args


def main():
    parser = _setup_parser()
    args = _check_args(parser)
    
    meta = nimare.meta.cbma.mkda.MKDAChi2()
    corrector = nimare.correct.FDRCorrector(alpha=0.05)
    
    data_dir = os.path.join(args.out_dir, 'concept_maps', 'neurosynth_files', args.vocab)
    concept_map_dir_uncorrected = os.path.join(args.out_dir, 'concept_maps', 'uncorrected')
    concept_map_dir_corrected = os.path.join(args.out_dir, 'concept_maps', 'corrected_fdr')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(concept_map_dir_uncorrected, exist_ok=True)
    os.makedirs(concept_map_dir_corrected, exist_ok=True)

    if os.path.exists(os.path.join(data_dir, 'neurosynth_dset.pkl.gz')):
        print("Neurosynth dataset already exists. Loading from file...")
        neurosynth_dset = nimare.dataset.Dataset.load(os.path.join(data_dir, 'neurosynth_dset.pkl.gz'))
    else:
        # Fetch Neurosynth with *just* the VOCAB features
        files = nimare.extract.fetch_neurosynth(
            data_dir=data_dir,  
            version="7",
            overwrite=False,
            source="abstract",
            vocab=args.vocab,  
        )
        neurosynth_db = files[0]

        # Get the Dataset object
        neurosynth_dset = nimare.io.convert_neurosynth_to_dataset(
            coordinates_file=neurosynth_db["coordinates"],
            metadata_file=neurosynth_db["metadata"],
            annotations_files=neurosynth_db["features"],
        )
        neurosynth_dset.annotations.head()
        neurosynth_dset.save(os.path.join(data_dir, 'neurosynth_dset.pkl.gz'))

    if os.path.exists(os.path.join(args.concept_mapping)):
        # Load the concept mapping dictionary
        concept_mapping_dictionary = json.loads(open(os.path.join(args.concept_mapping)).read())
    else:
        raise FileNotFoundError(f"Concept mapping file {args.concept_mapping} does not exist.")

    for lda_term, term in concept_mapping_dictionary.items():
        # Extract the term name (e.g., 'topic0', ..., 'topic49') for file naming

        # Define file paths for output
        corrected_map_file = f"{concept_map_dir_corrected}/{term}_z_desc-association_level-voxel_corr-FDR_method-indep.nii.gz"

        # Skip if both the association map and corrected map already exist
        if os.path.exists(corrected_map_file):
            continue

        # Get studies associated with the term
        term_ids = neurosynth_dset.get_studies_by_label(labels=lda_term, label_threshold=0.05)

        # Skip if no studies are found for this term
        if not term_ids:
            continue

        # Get studies that do NOT include this term
        all_ids = neurosynth_dset.ids
        notterm_ids = sorted(list(set(all_ids) - set(term_ids)))
        term_dset = neurosynth_dset.slice(term_ids)
        notterm_dset = neurosynth_dset.slice(notterm_ids)

        # Run the meta-analysis
        results = meta.fit(term_dset, notterm_dset)
        results.save_maps(output_dir=concept_map_dir_uncorrected, prefix=term, names=["z_desc-association"])

        # Correct for multiple comparisons
        results_corrected = corrector.transform(results)

        # Save the corrected map
        results_corrected.save_maps(output_dir=concept_map_dir_corrected, prefix=term, names=["z_desc-association_level-voxel_corr-FDR_method-indep"])

        # Add the term to the successful list

if __name__ == "__main__":
    main()