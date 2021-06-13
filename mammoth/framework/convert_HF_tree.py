"""Convert HF tree to parquet, making it easier to use.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""

import time
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd

from mammoth.framework import sources, utils


def hf_tree_to_parquet(filename: Path) -> bool:
    # Setup
    # First, we need the identifiers to group_by
    identifiers = [
        "run_number",
        # According to James:
        # TODO: Configure this, eventually...
        # - Data needs ev_id and ev_id_ext
        # - MC only needs ev_id
        "ev_id",
        #"ev_id_ext",
    ]
    # Particle columns and names.
    particle_level_columns = identifiers + [
        "ParticlePt",
        "ParticleEta",
        "ParticlePhi",
    ]
    # We want them to be stored in a standardized manner.
    _standardized_particle_names = {
        "ParticlePt": "pt",
        "ParticleEta": "eta",
        "ParticlePhi": "phi",
    }

    # Detector level
    det_level_tracks = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle",
        columns=particle_level_columns,
    )
    # Particle level
    part_level_tracks = sources.UprootSource(
        filename=filename,
        tree_name="PWGHF_TreeCreator/tree_Particle_gen",
        columns=particle_level_columns,
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

    # Convert the flat arrays into jagged arrays by grouping by the identifiers.
    # This allows us to work with the data as expected.
    det_level_tracks = utils.group_by(array=det_level_tracks.data(), by=identifiers)
    part_level_tracks = utils.group_by(array=part_level_tracks.data(), by=identifiers)
    # There is one entry per event, so we don't need to do any group by steps.
    event_properties = event_properties.data()

    # Event selection
    # We apply the event selection implicitly to the particles by requiring the identifiers
    # of the part level, det level, and event properties match.
    # NOTE: Since we're currently using this for conversion, it's better to have looser
    #       conditions so we can delete the original files. So we disable this for now.
    # event_properties = event_properties[
    #     (event_properties["is_ev_rej"] == 0)
    #     & (np.abs(event_properties["z_vtx_reco"]) < 10)
    # ]

    # Now, we're on to merging the particles and event level information. Remarkably, the part level,
    # det level, and event properties all have a unique set of identifiers. None of them are entirely
    # subsets of the others. This isn't particularly intuitive to me, but in any case, this seems to
    # match up with the way that it's handled in pyjetty.
    # If there future issues with merging, check out some additional thoughts and suggestions on
    # merging here: https://github.com/scikit-hep/awkward-1.0/discussions/633

    # First, grab the identifiers from each collection so we can match them up.
    # NOTE: For the tracks, everything is broadcasted to the shape of the particles, which is jagged,
    #       so we take the first instance for each event (since it will be the same for every particle
    #       in an event).
    det_level_tracks_identifiers = det_level_tracks[identifiers][:, 0]
    part_level_tracks_identifiers = part_level_tracks[identifiers][:, 0]
    event_properties_identifiers = event_properties[identifiers]

    # Next, find the overlap for each collection with each other collection, storing the result in
    # a mask.  As noted above, no collection appears to be a subset of the other.
    # Once we have the mask, we immediately apply it.
    # NOTE: isin doesn't work for a standard 2D array because a 2D array in the second argument will
    #       be flattened by numpy.  However, it works as expected if it's a structured array (which
    #       is the default approach for Array conversion, so we get a bit lucky here).
    det_level_tracks_mask = (
        np.isin(np.asarray(det_level_tracks_identifiers), np.asarray(part_level_tracks_identifiers))
        & np.isin(np.asarray(det_level_tracks_identifiers), np.asarray(event_properties_identifiers))
    )
    det_level_tracks = det_level_tracks[det_level_tracks_mask]
    part_level_tracks_mask = (
        np.isin(np.asarray(part_level_tracks_identifiers), np.asarray(det_level_tracks_identifiers))
        & np.isin(np.asarray(part_level_tracks_identifiers), np.asarray(event_properties_identifiers))
    )
    part_level_tracks = part_level_tracks[part_level_tracks_mask]
    event_properties_mask = (
        np.isin(np.asarray(event_properties_identifiers), np.asarray(det_level_tracks_identifiers))
        & np.isin(np.asarray(event_properties_identifiers), np.asarray(part_level_tracks_identifiers))
    )
    event_properties = event_properties[event_properties_mask]

    # Now, some rearranging the field names for uniformity.
    # Apparently, the array will simplify to associate the three fields together. I assumed that a zip
    # would be required, but apparently not.
    return ak.Array(
        {
            "det_level": ak.zip(
                dict(
                    zip(list(_standardized_particle_names.values()), ak.unzip(det_level_tracks[list(_standardized_particle_names.keys())]))
                )
            ),
            "part_level": ak.zip(
                dict(
                    zip(list(_standardized_particle_names.values()), ak.unzip(part_level_tracks[list(_standardized_particle_names.keys())]))
                )
            ),
            **dict(zip(ak.fields(event_properties), ak.unzip(event_properties))),
        },
    )


if __name__ == "__main__":
    #arrays = hf_tree_to_parquet(filename=Path("/software/rehlers/dev/substructure/trains/pythia/568/AnalysisResults.20g4.001.root"))
    arrays = hf_tree_to_parquet(filename=Path("/software/rehlers/dev/mammoth/AnalysisResults.root"))

    import IPython; IPython.embed()
