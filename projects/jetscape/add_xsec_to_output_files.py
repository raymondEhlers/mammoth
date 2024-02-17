""" Add jetscape cross section values to v1 FinalState* outputs

This could be brought in separately, but this is way more convenient and is pretty easy to do.

.. codeauthor:: Raymond Ehlers <raymond.ehlers@cern.ch>, ORNL
"""
from __future__ import annotations

from pathlib import Path


def add_xsec_to_output_files(output_file_dir: Path, output_filename_template: str, xsec_dir: Path) -> None:
    output_filenames = sorted(output_file_dir.glob(f"{output_filename_template}*"))

    for output_filename in output_filenames:
        # print(f"output filename: {output_filename}")

        # Extract pt hard bin
        pt_hat_bin = output_filename.name.replace(output_filename_template, "").replace(".out", "")
        # print(f"pt_hat_bin: {pt_hat_bin}")
        x_sec_filename = xsec_dir / f"SigmaHardBin{pt_hat_bin}.out"

        with x_sec_filename.open() as f:
            x_sec, x_sec_error = f.read().split()

        line_to_write = f"#\tsigmaGen\t{x_sec}\tsigmaErr\t{x_sec_error}"
        # print(f"line_to_write: {line_to_write}")

        print(f"Writing line to file '{output_filename}': {line_to_write}")
        with output_filename.open("a") as f:
            f.write(line_to_write)


if __name__ == "__main__":
    # For testing
    add_xsec_to_output_files(
        output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/test/"),
        output_filename_template="JetscapeHadronListBin",
        xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/test/"),
    )
    # pp
    # add_xsec_to_output_files(
    #    output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
    #    output_filename_template="JetscapeHadronListBin",
    #    xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/5020_PP_Colorless/"),
    # )
    # MATTER_LBT_RunningAlphaS_Q2qhat 0-5%
    # add_xsec_to_output_files(
    #    output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_0-5_0.30_2.0_1/"),
    #    output_filename_template="JetscapeHadronListBin",
    #    xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/pTHatCrossSection_5020GeV/"),
    # )
    ## MATTER_LBT_RunningAlphaS_Q2qhat 5-10%
    # add_xsec_to_output_files(
    #    output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_5-10_0.30_2.0_1/"),
    #    output_filename_template="JetscapeHadronListBin",
    #    xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/pTHatCrossSection_5020GeV/"),
    # )
    ## MATTER_LBT_RunningAlphaS_Q2qhat 30-40%
    # add_xsec_to_output_files(
    #    output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_30-40_0.30_2.0_1/"),
    #    output_filename_template="JetscapeHadronListBin",
    #    xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/pTHatCrossSection_5020GeV/"),
    # )
    ## MATTER_LBT_RunningAlphaS_Q2qhat 40-50%
    # add_xsec_to_output_files(
    #    output_file_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/MATTER_LBT_RunningAlphaS_Q2qhat/5020_PbPb_40-50_0.30_2.0_1/"),
    #    output_filename_template="JetscapeHadronListBin",
    #    xsec_dir=Path("/alf/data/rehlers/jetscape/osiris/AAPaperData/pTHatCrossSection_5020GeV/"),
    # )
