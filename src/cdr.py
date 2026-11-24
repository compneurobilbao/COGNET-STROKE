"""
Compute the Concept to Disconnection Representation.
This function gives a CDR value for a given conecpt map and disconnectivity map
and a p-value if nulls distributions of the CDR are provided."
"""

import numpy as np

def compute_cdr(concept_mask, disconection_map, nulls=None):
    """
    Compute the Concept to Disconnection Representation value and optional p-value.

    Parameters
    ----------
    concept_mask : numpy.ndarray
        Binary mask defining the concept regions (1: inside, -1: outside)
    disconection_map : numpy.ndarray
        Map of disconnection values
    nulls : numpy.ndarray, optional
        Null maps for statistical testing. If None, only CDR value is returned

    Returns
    -------
    representation_value : float
        The CDR value
    p_value : float or None
        The p-value if nulls were provided, None otherwise
    """
    inside_concept_mask = concept_mask == 1
    outside_concept_mask = concept_mask == -1
    
    inside_concept_dismap_mean = np.mean(disconection_map[inside_concept_mask])
    outside_concept_dismap_mean = np.mean(disconection_map[outside_concept_mask])
    if outside_concept_dismap_mean == 0:
        representation_value = inside_concept_dismap_mean
    else:
        representation_value = inside_concept_dismap_mean / outside_concept_dismap_mean

    # Early return if no nulls provided
    if nulls is None:
        return representation_value
    
    # Only execute this part if nulls are provided
    else:
        representation_value_nulls = []
        for null in nulls:
            inside_concept_mask_null = null == 1
            outside_concept_mask_null = null == -1

            inside_concept_dismap_mean_null = np.mean(disconection_map[inside_concept_mask_null])
            outside_concept_dismap_mean_null = np.mean(disconection_map[outside_concept_mask_null])
            if outside_concept_dismap_mean_null == 0:
                representation_value_nulls.append(inside_concept_dismap_mean_null)
            else:
                representation_value_nulls.append(inside_concept_dismap_mean_null / outside_concept_dismap_mean_null)

        representation_value_nulls = np.array(representation_value_nulls)
        number_of_hits = len(representation_value_nulls[representation_value_nulls > representation_value])
        if number_of_hits == 0:
            p_value = 1/len(representation_value_nulls)
        else:
            p_value = number_of_hits / len(representation_value_nulls)

        return representation_value, p_value

def compute_overlap(concept_mask, lesion_map, nulls=None):
    """
    Compute the overlap between lesion and concept and optional p-value.

    Parameters
    ----------
    concept_mask : numpy.ndarray
        Binary mask defining the concept regions (1: inside, -1: outside)
    lesion_map : numpy.ndarray
        Map of disconnection values
    nulls : numpy.ndarray, optional
        Null maps for statistical testing. If None, only CDR value is returned

    Returns
    -------
    representation_value : float
        The CDR value
    p_value : float or None
        The p-value if nulls were provided, None otherwise
    """
    inside_concept_mask = np.where(concept_mask == 1, 1, 0)
    
    overlap = np.sum(inside_concept_mask * lesion_map) / np.sum(lesion_map)

    # Early return if no nulls provided
    if nulls is None:
        return overlap
    
    # Only execute this part if nulls are provided
    else:
        overlap_value_nulls = []
        for null in nulls:
            inside_null_mask = np.where(null == 1, 1, 0)
            overlap_value_nulls.append(np.sum(inside_null_mask * lesion_map) / np.sum(lesion_map))

        overlap_value_nulls = np.array(overlap_value_nulls)
        number_of_hits = len(overlap_value_nulls[overlap_value_nulls > overlap])
        if number_of_hits == 0:
            p_value = 1/len(overlap_value_nulls)
        else:
            p_value = number_of_hits / len(overlap_value_nulls)

        return overlap, p_value