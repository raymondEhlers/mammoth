"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from mammoth.framework import sources, utils

def multi_dim_intersect(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    """Multi-dimensional intersection.

    From: https://stackoverflow.com/a/9271260/12907985

    """
    arr1_view = arr1.view([('',arr1.dtype)]*arr1.shape[1])
    arr2_view = arr2.view([('',arr2.dtype)]*arr2.shape[1])
    intersected = np.intersect1d(arr1_view, arr2_view)
    return intersected.view(arr1.dtype).reshape(-1, arr1.shape[1])

def structured_array_to_standard_array_view(arr: np.ndarray) -> np.ndarray:
    """View structured array as a standard array.

    Note: This requires that the dtypes of all of the fields must be the same!

    """
    return arr.view(arr.dtype[0]).reshape(-1, len(arr.dtype.names))


def hf_tree_to_parquet(filename: Path) -> bool:

    identifiers = [
        "run_number",
        # According to James:
        # TODO: Configure this, eventually...
        # - Data needs ev_id and ev_id_ext
        # - MC only needs ev_id
        "ev_id",
        #"ev_id_ext",
    ]

    track_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]

    det_level_tracks = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=track_level_columns,
    )
    # MC only
    part_level_tracks = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle_gen",
        columns=track_level_columns,
    )
    # Event level properties
    event_properties = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_event_char",
        columns=identifiers + [
            "z_vtx_reco", "is_ev_rej"
            # If it wasn't pythia, could also add:
            # - "centrality"
            # - event plane angle (but doesn't seem to be in HF tree output :-( )
        ],
    )

    det_level_tracks_data = det_level_tracks.data()
    part_level_tracks_data = part_level_tracks.data()
    event_properties_data = event_properties.data()

    #df_det_level_tracks = pd.DataFrame({k: np.asarray(v) for k, v in zip(ak.fields(det_level_tracks_data), ak.unzip(det_level_tracks_data))})
    #df_part_level_tracks = pd.DataFrame({k: np.asarray(v) for k, v in zip(ak.fields(part_level_tracks_data), ak.unzip(part_level_tracks_data))})
    #df_event_properties = pd.DataFrame({k: np.asarray(v) for k, v in zip(ak.fields(event_properties_data), ak.unzip(event_properties_data))})

    # Event selection
    #df_event_properties = df_event_properties[df_event_properties["is_ev_rej"] == 0]
    #df_event_properties.reset_index(drop=True)
    #full_df = pd.merge(df_det_level_tracks, df_event_properties, on=identifiers)
    # See; https://stackoverflow.com/a/47146609/12907985
    #full_df = pd.DataFrame().join([df_det_level_tracks, df_part_level_tracks, df_event_properties], on=identifiers)
    #full_df = pd.merge(df_det_level_tracks, df_event_properties, on=identifiers)

    # More ideas on merging here: https://github.com/scikit-hep/awkward-1.0/discussions/633

    #import IPython; IPython.embed()

    #det_level_tracks = utils.group_by(array=det_level_tracks.data(), by=identifiers)
    #part_level_tracks = utils.group_by(array=part_level_tracks.data(), by=identifiers)
    #event_properties = utils.group_by(array=event_properties.data(), by=identifiers)
    det_level_tracks = utils.group_by(array=det_level_tracks_data, by=identifiers)
    part_level_tracks = utils.group_by(array=part_level_tracks_data, by=identifiers)
    #event_properties = utils.group_by(array=event_properties_data, by=identifiers)
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = event_properties_data

    # Event selection
    event_properties = event_properties[
        (event_properties["is_ev_rej"] == 0)
        & (np.abs(event_properties["z_vtx_reco"]) < 10)
    ]

    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    #det_level_tracks_identifiers_arr = structured_array_to_standard_array_view(np.asarray(det_level_tracks_identifiers))
    part_level_tracks_identifiers = part_level_tracks[identifiers][:, 0]
    #part_level_tracks_identifiers_arr = structured_array_to_standard_array_view(np.asarray(part_level_tracks_identifiers))
    event_properties_identifiers = event_properties[identifiers]

    ## TODO: isin doesn't work because it flattens the second array...
    #res_is_in_det = np.isin(det_level_tracks_identifiers_arr, part_level_tracks_identifiers_arr, assume_unique=True)
    #res_combined_det = res_is_in_det[:, 0] & res_is_in_det[:, 1]
    #res_is_in_part = np.isin(part_level_tracks_identifiers_arr, det_level_tracks_identifiers_arr, assume_unique=True)
    #res_combined_part = res_is_in_part[:, 0] & res_is_in_part[:, 1]

    #new_det_level_tracks = det_level_tracks[res_combined_det]
    #new_part_level_tracks = part_level_tracks[res_combined_part]

    #import IPython; IPython.embed()
    print("About to calculate det level tracks mask")
    # NOTE: isin doesn't work for a standard 2D array because the 2D array in the second argument will be flattened by numpy.
    #       However, it works as expected if it's a structured array (which is the default approach for Array conversion
    det_level_tracks_mask = (
        np.isin(np.asarray(det_level_tracks_identifiers), np.asarray(part_level_tracks_identifiers))
        & np.isin(np.asarray(det_level_tracks_identifiers), np.asarray(event_properties_identifiers))
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    print("About to calculate part level tracks mask")
    part_level_tracks_mask = (
        np.isin(np.asarray(part_level_tracks_identifiers), np.asarray(det_level_tracks_identifiers))
        & np.isin(np.asarray(part_level_tracks_identifiers), np.asarray(event_properties_identifiers))
    )
    part_level_tracks = part_level_tracks[part_level_tracks_mask]
    print("About to calculate event properties mask")
    event_properties_mask = (
        np.isin(np.asarray(event_properties_identifiers), np.asarray(det_level_tracks_identifiers))
        & np.isin(np.asarray(event_properties_identifiers), np.asarray(part_level_tracks_identifiers))
    )
    event_properties = event_properties[event_properties_mask]
    print("Done!")

    # Now, some rearranging the field names for uniformity.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    #print("Making")
    #det_level = dict(
    #    zip(list(_standardized_particle_names.values()), ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]))
    #)
    #part_level = dict(
    #   zip(list(_standardized_particle_names.values()), ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]))
    #)
    #print("Done")

    #import IPython; IPython.embed()
    start = time.time()
    #zipped = ak.zip({
    #        "det_level": dict(
    #            zip(list(_standardized_particle_names.values()), ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]))
    #        ),
    #        "part_level": dict(
    #            zip(list(_standardized_particle_names.values()), ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]))
    #        ),
    #        "event": event_properties,
    #    },
    #    depth_limit=1,
    #)
    #zipped = ak.zip(
    #    {
    #        "det_level": det_level,
    #        "part_level": part_level,
    #        **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
    #    },
    #    depth_limit = 1,
    #)

    #arrays = ak.Array(
    #    {
    #        "det_level": det_level,
    #        "part_level": part_level,
    #        **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
    #    },
    #)
    arrays = ak.Array(
        {
            "det_level": dict(
                zip(list(_standardized_particle_names.values()), ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]))
            ),
            "part_level": dict(
               zip(list(_standardized_particle_names.values()), ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]))
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )
    print(f"zip time: {time.time() - start}")
    return arrays


if __name__ == "__main__":
    #arrays = hf_tree_to_parquet(filename=Path("/software/rehlers/dev/substructure/trains/pythia/568/AnalysisResults.20g4.001.root"))
    arrays = hf_tree_to_parquet(filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.root"))
    print("Returned")

    import IPython; IPython.embed()
