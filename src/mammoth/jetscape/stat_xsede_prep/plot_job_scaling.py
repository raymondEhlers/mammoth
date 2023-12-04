
""" Plot Stampede2 scaling information.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch> ORNL
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pachyderm.plot

pachyderm.plot.configure()
# Enable ticks on all sides
# Unfortunately, some of this is overriding the pachyderm plotting style.
# That will have to be updated eventually...
mpl.rcParams["xtick.top"] = True
mpl.rcParams["xtick.minor.top"] = True
mpl.rcParams["ytick.right"] = True
mpl.rcParams["ytick.minor.right"] = True

def error_prop_divide(num_value: np.float64, num_delta: np.float64, denom_value: np.float64, denom_delta: np.float64) -> tuple[np.float64, np.float64]:
    val = num_value / denom_value
    delta = val * np.sqrt((num_delta / num_value) ** 2 + (denom_delta / denom_value) ** 2)
    return (val, delta)


def determine_scaling_values(scaling_info: Mapping[str, Mapping[str, Sequence[float]]]) -> tuple[
    dict[int, tuple[np.float64, np.float64]],
    dict[int, tuple[np.float64, np.float64, np.float64, np.float64]]
]:
    model = "matter_lbt"

    values: dict[int, tuple[np.float64, np.float64]] = {}
    # Extract the values
    for n_cores in [1, 6, 20, 36, 42, 48]:
        # This is the runtime in seconds.
        arr = np.array(scaling_info[f"skylake_{n_cores}_seed_1_1"][model])
        # Convert to events/hour using 600 events
        arr = 600 / (arr / 60) * 60
        # Finally, store the mean and std deviation.
        values[n_cores] = (np.mean(arr), np.std(arr))

    # Normalize the values to 1 core.
    normalized_values = {}
    base_mean, base_delta = values[1]
    for k, (mean, delta) in values.items():
        val, delta_val = error_prop_divide(mean, delta, base_mean, base_delta)
        _, delta_stochastic_5 = error_prop_divide(mean, mean * 0.05, base_mean, base_mean * 0.05)
        _, delta_stochastic_10 = error_prop_divide(mean, mean * 0.1, base_mean, base_mean * 0.1)
        normalized_values[k] = (val, delta_val, delta_stochastic_5, delta_stochastic_10)

    return values, normalized_values


def plot(scaling: Mapping[int, tuple[np.float64, np.float64, np.float64, np.float64]]) -> None:
    # Extract values
    x_values = list(scaling.keys())
    y_values = [v[0] for v in scaling.values()]
    y_stat_error = [v[1] for v in scaling.values()]  # noqa: F841
    y_stochastic_5 = np.array([v[2] for v in scaling.values()])
    y_stochastic_10 = np.array([v[3] for v in scaling.values()])  # noqa: F841

    fig, ax = plt.subplots(figsize=(8, 6))

    data = ax.errorbar(
        x_values, y_values,
        # Skip showing these error bars...
        #yerr=y_stat_error,
        marker=".",
        markersize=11,
        linestyle="",
        color="black",
        zorder=10,
        label="Mean",
    )

    # Basically just need an error box, so we have to give it some x width...
    x_error = np.array([1] * len(x_values))
    pachyderm.plot.error_boxes(
        ax=ax,
        x_data=x_values,  # type: ignore[arg-type]
        y_data=y_values,  # type: ignore[arg-type]
        x_errors=x_error,
        y_errors=y_stochastic_5,
        color="green",
        linewidth=0,
        label=r"Stochastic 5\% error",
        zorder=6,
    )
    #error_10 = pachyderm.plot.error_boxes(
    #    ax=ax,
    #    x_data=x_values,
    #    y_data=y_values,
    #    x_errors=x_error,
    #    y_errors=y_stochastic_10,
    #    color="blue",
    #    linewidth=0,
    #    label="Stochastic 10\% error",
    #    zorder=5,
    #)

    ax.axhline(y=1, color="black", linestyle="dashed", zorder=1)
    ax.set_xlim([0, 50])
    ax.set_ylim([0, 1.5])

    ax.set_xlabel("$N$ Skylake cores")
    ax.set_ylabel("Event rate compared to $N=1$")

    legend_info = [
        data,
        mpatches.Patch(color='green', alpha=0.5, label=r"Stochastic 5\% error"),
        #mpatches.Patch(color='blue', alpha=0.5, label="Stochastic 10\% error"),
    ]

    ax.legend(
        handles=legend_info,
        loc="lower left",
        frameon=False,
    )

    fig.tight_layout()
    fig.savefig("matter_lbt_5020_single_hard_trigger_scaling.pdf")


if __name__ == "__main__":
    single_hard_trigger_5020_scaling_info = {
        'skylake_1_seed_1_1': {'matter_lbt': [2361.0]},
        'skylake_20_seed_1_1': {'matter_lbt': [2532.0,
                                               2335.0,
                                               2578.0,
                                               2375.0,
                                               2592.0,
                                               2346.0,
                                               2369.0,
                                               2596.0,
                                               2584.0,
                                               2370.0,
                                               2371.0,
                                               2598.0,
                                               2583.0,
                                               2379.0,
                                               2587.0,
                                               2366.0,
                                               2385.0,
                                               2607.0,
                                               2381.0,
                                               2579.0]},
        'skylake_36_seed_1_1': {'matter_lbt': [2672.0,
                                               2577.0,
                                               2718.0,
                                               2685.0,
                                               2678.0,
                                               2694.0,
                                               2670.0,
                                               2691.0,
                                               2675.0,
                                               2703.0,
                                               2658.0,
                                               2734.0,
                                               2649.0,
                                               2657.0,
                                               2718.0,
                                               2665.0,
                                               2666.0,
                                               2700.0,
                                               2720.0,
                                               2655.0,
                                               2710.0,
                                               2673.0,
                                               2713.0,
                                               2706.0,
                                               2692.0,
                                               2654.0,
                                               2663.0,
                                               2700.0,
                                               2695.0,
                                               2665.0,
                                               2721.0,
                                               2647.0,
                                               2654.0,
                                               2685.0,
                                               2708.0,
                                               2692.0]},
        'skylake_42_seed_1_1': {'matter_lbt': [2728.0,
                                               2747.0,
                                               2792.0,
                                               2806.0,
                                               2774.0,
                                               2821.0,
                                               2805.0,
                                               2802.0,
                                               2808.0,
                                               2808.0,
                                               2822.0,
                                               2814.0,
                                               2783.0,
                                               2795.0,
                                               2819.0,
                                               2837.0,
                                               2825.0,
                                               2788.0,
                                               2823.0,
                                               2832.0,
                                               2807.0,
                                               2854.0,
                                               2819.0,
                                               2823.0,
                                               2807.0,
                                               2824.0,
                                               2815.0,
                                               2820.0,
                                               2803.0,
                                               2829.0,
                                               2822.0,
                                               2795.0,
                                               2800.0,
                                               2804.0,
                                               2798.0,
                                               2818.0,
                                               2792.0,
                                               2778.0,
                                               2796.0,
                                               2831.0,
                                               2806.0,
                                               2811.0]},
        'skylake_48_seed_1_1': {'matter_lbt': [2824.0,
                                               2796.0,
                                               2868.0,
                                               2921.0,
                                               2910.0,
                                               2944.0,
                                               2897.0,
                                               2928.0,
                                               2889.0,
                                               2934.0,
                                               2884.0,
                                               2911.0,
                                               2837.0,
                                               2906.0,
                                               2912.0,
                                               2938.0,
                                               2907.0,
                                               2895.0,
                                               2957.0,
                                               2931.0,
                                               2918.0,
                                               2873.0,
                                               2930.0,
                                               2852.0,
                                               2890.0,
                                               2926.0,
                                               2897.0,
                                               2959.0,
                                               2925.0,
                                               2867.0,
                                               2890.0,
                                               2887.0,
                                               2915.0,
                                               2911.0,
                                               2879.0,
                                               2931.0,
                                               2853.0,
                                               2869.0,
                                               2915.0,
                                               2896.0,
                                               2868.0,
                                               2864.0,
                                               2942.0,
                                               2879.0,
                                               2916.0,
                                               2875.0,
                                               2908.0,
                                               2874.0]},
        'skylake_6_seed_1_1': {'matter_lbt': [2234.0,
                                              2265.0,
                                              2291.0,
                                              2278.0,
                                              2303.0,
                                              2288.0]}
    }

    res = determine_scaling_values(single_hard_trigger_5020_scaling_info)
    plot(res[1])
