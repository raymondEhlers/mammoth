"""Tools for comparison flat substructure outputs

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL
"""

import logging
import pprint
import warnings
from pathlib import Path
from typing import Optional, Sequence

import attrs
import awkward as ak
import hist
import mammoth.helpers
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot
import uproot
from pachyderm import binned_data, plot as pb

pachyderm.plot.configure()


logger = logging.getLogger(__name__)


@attrs.define
class Input:
    name: str = attrs.field()
    arrays: ak.Array = attrs.field()
    attribute: str = attrs.field()


def arrays_to_hist(
    arrays: ak.Array, attribute: str, axis: hist.axis.Regular = hist.axis.Regular(30, 0, 150)
) -> binned_data.BinnedData:
    h_hist = hist.Hist(axis, storage=hist.storage.Weight())
    h_hist.fill(ak.flatten(arrays[attribute], axis=None))

    return binned_data.BinnedData.from_existing_data(h_hist)

def compare_branch(standard: ak.Array, track_skim: ak.Array, key: str, variable_name: str, assert_false_on_failed_comparison: bool = False) -> bool:
    # Setup
    success = True
    standard_array = standard[key]
    track_skim_array = track_skim[key]

    try:
        is_array_all_close = np.allclose(ak.to_numpy(standard_array), ak.to_numpy(track_skim_array), rtol=1e-4)
        logger.info(f"{variable_name}  all close? {is_array_all_close}")
        if not is_array_all_close:
            _arr = ak.zip({"s": standard_array, "t": track_skim_array})
            logger.info(pprint.pformat(_arr.to_list()))
            is_not_close_array = np.where(~np.isclose(ak.to_numpy(standard_array), ak.to_numpy(track_skim_array)))
            logger.info(f"Indices where not close: {is_not_close_array}")
            success = False
    except ValueError as e:
        logger.exception(e)
        success = False
        if assert_false_on_failed_comparison:
            assert False

    # If the above failed, print the entire branch.
    # Sometimes it's useful to start at this, but sometimes it's just overwhelming, so uncomment as necessary
    if not success:
        logger.info("Values from above:")
        logger.info(f"standard_{variable_name}: {standard_array.to_list()}")
        logger.info(f"track_skim_{variable_name}: {track_skim_array.to_list()}")
        logger.info("**************************")

    return success


def plot_attribute_compare(
    other: Input,
    mine: Input,
    plot_config: pb.PlotConfig,
    output_dir: Path,
    axis: hist.axis.Regular = hist.axis.Regular(30, 0, 150),
    normalize: bool = False,
) -> None:
    # Plot
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    other_hist = arrays_to_hist(arrays=other.arrays, attribute=other.attribute, axis=axis)
    mine_hist = arrays_to_hist(arrays=mine.arrays, attribute=mine.attribute, axis=axis)
    # Normalize
    if normalize:
        other_hist /= np.sum(other_hist.values)
        mine_hist /= np.sum(mine_hist.values)

    ax.errorbar(
        other_hist.axes[0].bin_centers,
        other_hist.values,
        xerr=other_hist.axes[0].bin_widths / 2,
        yerr=other_hist.errors,
        label=other.name,
        linestyle="",
        alpha=0.8,
    )
    ax.errorbar(
        mine_hist.axes[0].bin_centers,
        mine_hist.values,
        xerr=mine_hist.axes[0].bin_widths / 2,
        yerr=mine_hist.errors,
        label=mine.name,
        linestyle="",
        alpha=0.8,
    )

    ratio = mine_hist / other_hist
    ax_ratio.errorbar(
        ratio.axes[0].bin_centers, ratio.values, xerr=ratio.axes[0].bin_widths / 2, yerr=ratio.errors, linestyle=""
    )
    logger.info(f"ratio sum: {np.sum(ratio.values)}")
    logger.info(f"other: {np.sum(other_hist.values)}")
    logger.info(f"mine: {np.sum(mine_hist.values)}")

    # Apply the PlotConfig
    plot_config.apply(fig=fig, axes=[ax, ax_ratio])

    # filename = f"{plot_config.name}_{jet_pt_bin}{grooming_methods_filename_label}_{identifiers}_iterative_splittings"
    filename = f"{plot_config.name}"
    fig.savefig(output_dir / f"{filename}.pdf")
    plt.close(fig)


def _pretty_print_flat_type(s: str) -> str:
    """ Dumb pretty print function.

    It uses no logic, but it's fairly convenient for flat trees.
    """
    return "\n".join(s.split(","))


def compare_flat_substructure(  # noqa: C901
    collision_system: str,
    jet_R: float,
    prefixes: Sequence[str],
    standard_filename: Path,
    track_skim_filename: Path,
    standard_tree_name: str = "tree",
    base_output_dir: Path = Path("comparison/track_skim"),
    track_skim_validation_mode: bool = True,
    assert_false_on_failed_comparison_for_debugging_during_testing: bool = False,
) -> bool:
    """ Compare flat substructure productions

    Args:
        ...
        assert_false_on_failed_comparison_for_debugging_during_testing: If True, assert False
            when a comparison fails. This is useful during tests because we can run pytest with
            the `--pdb` option, which will automatically open a debugger at that failed line so
            that we can investigate further. The test will fail later since we return False on
            a failed comparison, but this is more convenient since it allows us to immediately
            access the underlying arrays. Default: False.
    """
    standard = uproot.open(standard_filename)[standard_tree_name].arrays()
    track_skim = uproot.open(track_skim_filename)["tree"].arrays()
    # Display the types for convenience in making the comparison
    logger.info(f"standard.type: {_pretty_print_flat_type(str(standard.type))}")
    logger.info(f"track_skim.type: {_pretty_print_flat_type(str(track_skim.type))}")

    # For the track skim validation:
    # - For whatever reason, the sorting of the jets is inconsistent for some collision_system + R (but not all).
    #   It seems to happen when there are multiple jets in one event, although I can't pin it down precisely.
    # - As a pragmatic approach, when doing the track skim validation, we allow the track skim to be reordered.
    #   We have the tools here to determine the new order automatically based on the jet pt (since it's unlikely that
    #   two jets will have precisely the same jet pt in our sample).
    # - Once the new order has been determined, we can store it in the map here so that we don't continue to emit
    #   warnings for expected differences.
    # - So we just apply a mask here to swap the one event where we have two jets.
    # NOTE: It appears that AliPhysics is actually the one that gets the sorting wrong here...
    # NOTE: This is a super specialized thing for the validation, but better to do it here instead of messing around
    #       with the actual mammoth analysis code.
    if track_skim_validation_mode:
        # First, we deal with the known reorder map.
        # If there are entries here, then we'll use them.
        # The keys are (collision_system, jet_R)
        reorder_map = {
            # Indices of differences: [15, 16]
            ("embed_pythia", 0.4): [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 15, 17],
            # Indices of differences: [1, 2, 4, 5, 14, 15, 19, 21, 25, 26, 28, 29]
            ("embed_pythia", 0.2): [0, 2, 1, 3, 5, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 14, 16, 17, 18, 21, 20, 19, 22, 23, 24, 26, 25, 27, 29, 28],
            # Indices of differences: [6, 7]
            ("PbPb", 0.2): [0, 1, 2, 3, 4, 5, 7, 6],
        }

        if (collision_system, jet_R) in reorder_map:
            # First, verify that they're the same length. If they're not, we don't want to mess with the mask
            if len(track_skim) != len(standard):
                warning_for_user = (
                    "Cannot use predefined mask because track_skim and standard aren't the same length."
                    f" len(standard)={len(standard)}, len(track_skim)={len(track_skim)}"
                )
                # log it in the correct place for convenience in understanding the logs
                logger.warning(warning_for_user)
                # And further emit it in case the test passes so the user doesn't overlook this
                warnings.warn(UserWarning(warning_for_user))
            else:
                reorder_mask = reorder_map[(collision_system, jet_R)]
                track_skim = track_skim[reorder_mask]

        # Take a peek with the jet pt, which should be easy to match (and unlikely to have exact duplicates).
        # This will allow us to determine the order (if it needs to be reordered)
        _prefix = prefixes[0]
        result = compare_branch(
            standard=standard, track_skim=track_skim, key=f"{_prefix}_jet_pt", variable_name="jet_pt",
            assert_false_on_failed_comparison=assert_false_on_failed_comparison_for_debugging_during_testing,
        )
        # They disagree. We'll try to figure out if it's just a minor ordering issue.
        if not result:
            try:
                # Describe the indices where there are disagreements
                is_not_close_array = np.where(~np.isclose(ak.to_numpy(standard[f"{_prefix}_jet_pt"]), ak.to_numpy(track_skim[f"{_prefix}_jet_pt"])))

                # To get the same indexing, we want to go:
                # track_skim -> sorted track_skim (if same values, it's the same order as sorted standard)
                # -> undo argsort of standard.
                # To undo the argsort of the standard, we will argsort the argsort output.
                # Think about it a while, and it makes sense. See also: https://stackoverflow.com/a/54799987/12907985
                standard_arg_sort = ak.argsort(standard[f"{_prefix}_jet_pt"])
                # NOTE: Even if you want to use ascending=False for the jet pt since it's conceptually nice to
                #       think of those in descending order, we _don't_ use ascending=False for the undo step
                #       since this argsort is working with indices, so we want ascending.
                #       The ascending vs descending arguments only need to be symmetric for the jet_pt sort
                undo_standard_arg_sort = ak.argsort(standard_arg_sort)
                reorder_mask = ak.argsort(track_skim[f"{_prefix}_jet_pt"])[undo_standard_arg_sort]

                warning_for_user = (
                    "Order of standard and track_skim arrays appears to be different. Will attempt to put them in the same order."
                    " Careful to check that this isn't somehow meaningful!"
                    f"\nIndices of differences: {ak.to_list(is_not_close_array[0])}."
                    f"\nNew mask: {ak.to_list(reorder_mask)}"
                )
                # log it in the correct place for convenience in understanding the logs
                logger.warning(warning_for_user)
                # And further emit it in case the test passes so the user doesn't overlook this
                warnings.warn(UserWarning(warning_for_user))
                track_skim = track_skim[reorder_mask]
            except ValueError:
                # If this fails, it's probably because the arrays are different lengths.
                # In that case, it's usually best to see the other values to help try to
                # sort it out. So continue the comparison instead of stopping here.
                pass

    output_dir = base_output_dir / f"{collision_system}__jet_R{round(jet_R*100):03}"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_success = True
    for prefix in prefixes:
        logger.info(f"Comparing prefix '{prefix}'")

        text = f"{collision_system.replace('_', ' ')}: {prefix.replace('_', ' ')}"
        plot_attribute_compare(
            other=Input(arrays=standard, attribute=f"{prefix}_jet_pt", name="Standard"),
            mine=Input(arrays=track_skim, attribute=f"{prefix}_jet_pt", name="Track skim"),
            plot_config=pb.PlotConfig(
                name=f"{prefix}_jet_pt",
                panels=[
                    # Main panel
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "y",
                                label="Prob.",
                                log=True,
                                font_size=22,
                            ),
                        ],
                        text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                        legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                    ),
                    # Data ratio
                    pb.Panel(
                        axes=[
                            pb.AxisConfig(
                                "x",
                                label=r"$p_{\text{T,ch jet}}$ (GeV/$c$)",
                                font_size=22,
                            ),
                            pb.AxisConfig(
                                "y",
                                label=r"Track skim/Standard",
                                range=(0.6, 1.4),
                                font_size=22,
                            ),
                        ],
                    ),
                ],
                figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
            ),
            output_dir=output_dir,
            axis=hist.axis.Regular(50, 0, 100),
            normalize=True,
        )
        result = compare_branch(
            standard=standard, track_skim=track_skim, key=f"{prefix}_jet_pt", variable_name="jet_pt",
            assert_false_on_failed_comparison=assert_false_on_failed_comparison_for_debugging_during_testing,
        )
        # We only want to assign the result if it's false because we don't want to accidentally overwrite
        # a failure with a success at the end
        if not result:
            all_success = result

        for grooming_method in ["dynamical_kt", "soft_drop_z_cut_02"]:
            logger.info(f'Plotting method "{grooming_method}"')
            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_kt", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_kt", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_kt",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="center right", anchor=(0.985, 0.52), font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$k_{\text{T,g}}$ (GeV/$c$)",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=hist.axis.Regular(50, 0, 10),
                output_dir=output_dir,
            )

            result = compare_branch(
                standard=standard, track_skim=track_skim, key=f"{grooming_method}_{prefix}_kt", variable_name="kt",
                assert_false_on_failed_comparison=assert_false_on_failed_comparison_for_debugging_during_testing,
            )
            # We only want to assign the result if it's false because we don't want to accidentally overwrite
            # a failure with a success at the end
            if not result:
                all_success = result

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_delta_R", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_delta_R", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_delta_R",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$R_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                output_dir=output_dir,
                axis=hist.axis.Regular(50, 0, 0.6),
                normalize=True,
            )

            result = compare_branch(
                standard=standard, track_skim=track_skim, key=f"{grooming_method}_{prefix}_delta_R", variable_name="delta_R",
                assert_false_on_failed_comparison=assert_false_on_failed_comparison_for_debugging_during_testing,
            )
            # We only want to assign the result if it's false because we don't want to accidentally overwrite
            # a failure with a success at the end
            if not result:
                all_success = result

            plot_attribute_compare(
                other=Input(arrays=standard, attribute=f"{grooming_method}_{prefix}_z", name="Standard"),
                mine=Input(arrays=track_skim, attribute=f"{grooming_method}_{prefix}_z", name="Track skim"),
                plot_config=pb.PlotConfig(
                    name=f"{grooming_method}_{prefix}_z",
                    panels=[
                        # Main panel
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "y",
                                    label="Prob.",
                                    log=True,
                                    font_size=22,
                                ),
                            ],
                            text=pb.TextConfig(x=0.97, y=0.97, text=text, font_size=22),
                            legend=pb.LegendConfig(location="upper left", font_size=22),
                        ),
                        # Data ratio
                        pb.Panel(
                            axes=[
                                pb.AxisConfig(
                                    "x",
                                    label=r"$z_{\text{g}}$",
                                    font_size=22,
                                ),
                                pb.AxisConfig(
                                    "y",
                                    label=r"Track skim/Standard",
                                    range=(0.6, 1.4),
                                    font_size=22,
                                ),
                            ],
                        ),
                    ],
                    figure=pb.Figure(edge_padding=dict(left=0.13, bottom=0.115)),
                ),
                normalize=True,
                axis=hist.axis.Regular(50, 0, 0.5),
                output_dir=output_dir,
            )

            result = compare_branch(
                standard=standard, track_skim=track_skim, key=f"{grooming_method}_{prefix}_z", variable_name="zg",
                assert_false_on_failed_comparison=assert_false_on_failed_comparison_for_debugging_during_testing,
            )
            # We only want to assign the result if it's false because we don't want to accidentally overwrite
            # a failure with a success at the end
            if not result:
                all_success = result

    return all_success


def run(jet_R: float, collision_system: str, prefixes: Optional[Sequence[str]] = None) -> None:
    """Trivial helper for running the comparison.

    It's not very configurable, but it provides a reasonable example.
    """
    if prefixes is None:
        prefixes = ["data"]
    mammoth.helpers.setup_logging()
    logger.info(f"Running {collision_system} with prefixes {prefixes}")
    path_to_mammoth = Path(mammoth.helpers.__file__).parent.parent
    standard_base_filename = "AnalysisResults"
    if collision_system == "pythia":
        standard_base_filename += ".12"
    compare_flat_substructure(
        collision_system=collision_system,
        jet_R=jet_R,
        prefixes=prefixes,
        standard_filename=path_to_mammoth
        / f"projects/framework/{collision_system}/1/skim/{standard_base_filename}.repaired.00_iterative_splittings.root",
        track_skim_filename=path_to_mammoth / f"projects/framework/{collision_system}/1/skim/skim_output.root",
    )
