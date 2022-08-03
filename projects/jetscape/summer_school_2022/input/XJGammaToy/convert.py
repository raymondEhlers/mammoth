#!/usr/bin/env python3

from typing import Tuple, Mapping

import numpy as np
import numpy.typing as npt
from pachyderm import binned_data
import uproot

def get_data(centrality: str, normalize: bool) -> Tuple[binned_data.BinnedData, binned_data.BinnedData, binned_data.BinnedData]:
    centrality_index_map = {
        "50-100": 1,
        # pp smeared with 50-100
        "50-100pp": 2,
        "30-50": 3,
        # pp smeared with 30-50
        "30-50pp": 4,
        "10-30": 5,
        # pp smeared with 10-30
        "10-30pp": 6,
        "0-10": 7,
        # pp smeared with 0-10
        "0-10pp": 8,
    }
    cent_index = centrality_index_map[centrality]

    with uproot.open("HEPData-ins1638996-v1-Table_6.root") as f:
        data = binned_data.BinnedData.from_existing_data(f["Table 6"][f"Hist1D_y{cent_index}"])
        stat_error = binned_data.BinnedData.from_existing_data(f["Table 6"][f"Hist1D_y{cent_index}_e1"])
        sys_error = binned_data.BinnedData.from_existing_data(f["Table 6"][f"Hist1D_y{cent_index}_e2"])

    val = np.sum(data.values)
    if normalize:
        print(f"integral: {val}")
        data /= val
        stat_error /= val
        sys_error /= val

    return data, stat_error, sys_error

def run(centrality_to_filename: Mapping[Tuple[int, int], str], normalize: bool) -> None:
    all_data = {
        "50-100": get_data("50-100", normalize=normalize),
        "30-50": get_data("30-50", normalize=normalize),
        "10-30": get_data("10-30", normalize=normalize),
        "0-10": get_data("0-10", normalize=normalize),
    }

    for centrality_range, filename in centrality_to_filename.items():
        name = f"{centrality_range[0]}-{centrality_range[1]}"
        # Unpack loaded data
        data, stat_error, sys_error = all_data[name]

        with open(filename, "w") as f:
            print(name)
            # First, write the header
            f.write(f"""# Version 1.0
# DOI
# Source
# System PbPb5020
# Centrality {centrality_range[0]} {centrality_range[1]}
# XY XJGamma
# Label x y stat,low stat,high sys,low sys,high
""")
            # 50-100 is missing the first point. So we fill in an empty point for it at 0.0625
            if name == "50-100":
                values_to_write = np.array([(0.0625, 0.0, 0.0, 0.0, 0.0, 0.0)])
                #f.write(str(values_to_write))
                np.savetxt(f, values_to_write, fmt="%f")
            values_to_write = np.array([
                data.axes[0].bin_centers, data.values, stat_error.errors, stat_error.errors, sys_error.errors, sys_error.errors,
            ]).T
            #for bin_center, value, stat, sys in zip(data.axes[0].bin_centers, data.values, stat_error.errors, sys_error.errors):
            #    values_to_write = (bin_center, value, stat, stat, sys, sys)
            np.savetxt(f, values_to_write, fmt="%f")
            #f.write(str(values_to_write))

if __name__ == "__main__":
    run(
        centrality_to_filename={
            (0, 10): "Data_Selection1.dat",
            (50, 100): "Data_Selection2.dat",
        },
        # Since we generate a toy distribution, we can't know the overall normalization. Much easier to just normalize the data.
        normalize=True,
    )
