# Running an analysis

After [developing an analysis](developing_an_analysis.md), you need to follow some conventions to successfully take advantage of the analysis framework[^1]. Below, we'll use an example analysis to guide through this process.

[^1]: For clarity, if you have some separately developed piece of code, you don't necessarily need to run it through the analysis framework. The framework is designed to help you run your analysis (particularly in parallel), but you can also go it alone. If you decide to go that route, I'd suggest that you clearly label that code as such.

> [!tip]
> I've tried to document the main concepts below, but the bottom line is this documentation cannot cover everything at this time. If you're trying to do something, I recommend searching around the repository. What you're looking for has probably already been done for some analysis, and so you can adapt that existing example for your own purposes.

# Key concepts

Each run of an analysis is known as a "production". This corresponds to a single run of a given analysis code over a single dataset with a specified set of analysis parameters. For example, this could be analysis of groomed jet Rg over the ALICE 2024 pp reference dataset for R = 0.4 jets, soft drop z_cut = 0.2, etc... ([We cover analysis parameters below](#analysis-settings), but for now, let's stick to higher level run parameters).

A production is defined in the `track_skim_config.yaml` defined in `src/mammoth/alice/config/track_skim_config.yaml`. There are a few different types of entries:

1. Datasets
2. General analysis parameters
3. Collision system specific productions

> [!tip]
> The yaml config is stored in the ALICE directory for historical reasons, but you can put any analysis in there. There can be cases where it might be nice to put it elsewhere - to do so, just adjust the `config_filename` argument that you pass to your production definition.

The possible collision systems are:

- pp: pp data
- pp_MC: pp MC. Could be either both track + particle level, or you can select just one.
- PbPb: PbPb data
- PbPb_MC: PbPb MC. Could be either both track + particle level, or you can select just one.
- embed_pythia: pp MC embedded into real PbPb data (n.b. this should be understood as embed_pp_MC, but we've only ever used pythia, so I've kept the historical name).
- embed_thermal_model: pp MC embedded into a thermal background, generated according to specified parameters.

> [!note]
> All productions are defined in the same file, even if they're running for different analyses. This ensures that the production numbers are unique.

> [!tip]
> For ALICE folks, you can imagine this as an analogue to the LEGO train / Hyperloop. The idea is to track each run of an analysis, so we always know how it was used

## Datasets

A dataset corresponds to one input source, such as a track skim. They are stored under the `skimmed_datasets` key, and are named by the LHC run period that they correspond to. As an example, I've documented the LHC24 pp_ref dataset definition below to describe the fields:

```yaml
# LHC24_pp_ref, including LHC24ap and LHC24aq
LHC24_pp_ref_pass1_AOD_626869: &LHC24_pp_ref_pass1 # Name of the dataset. Should match the key
  name: "LHC24_pp_ref_pass1_AOD_626869"
  # The collision system of the dataset
  collision_system: "pp"
  # sqrt_s
  sqrt_s: 5360
  # LHC run period
  # This is not externally validated, so just put in what makes sense.
  # For example, the pp_ref is LHC24ap and LHC24aq, so I just merge those together.
  period: "LHC24apaq"
  # How the skim was created. This should match the name defined in the io module,
  # and will be used to determine how to read the skim
  skim_type: "run3_berkeley_skim"
  # Parameters that are relevant to the skim. These vary between each skim. Check the
  # definition of the skim reader in the io module for information on what is accepted for each
  skim_parameters:
    version: 2
  # Note if run 3 or not
  run3: true
  # Tree name of the input files
  tree_name: "eventTree"
  # Name of the input track "level(s)". By convention, one level is usually labeled as "data".
  # There could also be multiple levels - e.g. particle and detector level in pp/PbPb_MC productions.
  # The purpose of this list to indicate what should be read from the file (i.e. this could be used
  # to skip reading a level we may not be interested in)
  levels_in_output_file:
    - "data"
  # Number of parent directories to go up to create the relative output filename
  # This is used as an identifier in e.g. the output files. We want to go up enough levels
  # that the identifier is unique, but not go so far that reading the identifier is unwieldy.
  # Usually this is determined by trial and error
  number_of_parent_directories_for_relative_output_filename: 2
  # Specify where the track skim files are stored. There are two options (which must be used uniquely - no mixing):
  # 1) Specify the paths directly, using wildcards as needed
  # 2) Specify a file list ending in .txt, which contains the path to all of the files (e.g. extracted using `find`)
  # In either case, you can specify multiple list - the values that are extracted from each list entry will be combined together
  files:
    - "pp/626869/files.txt"
```

If you're trying to add a new dataset, I recommend you look for an example of an existing dataset of the same type, and adapt from that case.

### Dataset workflow

Below is a quick overview on how to produce ALICE run 2 and run 3 track skims:

- Run 2: AliAnalysisTaskTrackSkim -> Mammoth analysis
  - Run the AliPhysics analysis task [PWGJE/EMCALJetTasks/UserTasks/AliAnalysisTaskTrackSkim.cxx](https://github.com/alisw/AliPhysics/blob/master/PWGJE/EMCALJetTasks/UserTasks/AliAnalysisTaskTrackSkim.cxx) on the LEGO train.
  - Download the outputs - they're immediately read to use in mammoth!
- Run 3: ALICE AO2D -> JE derived data -> "Berkeley skim" -> Mammoth analysis
  - ALICE AO2D: Default AOD file produced by the collaboration
  - JE Derived Data: Tables produced in the ALICE JE data model, such as `JCollisions`, `JTracks`, .... These are produced for any JE analysis run in O2. For our workflow, the output from the derived data model should be saved. NOTE: Unless you're producing the derived data for the entire JE (rare), this output should not be marked as data to be stored for a derived dataset.
  - Next, information is extracted from the JE derived dataset output to create a so-called "Berkeley Tree". You need to access and download the files where the derived datasets are stored - these can be found in hyperloop.
    - The steps from downloading through conversion are handled by the [AO2DToBerkeleyTreeConverter](https://github.com/KirillVNaumov/AO2DToBerkeleyTreeConverter). These is documentation on its use [here](https://github.com/KirillVNaumov/AO2DToBerkeleyTreeConverter/blob/master/scripts/README.md).
    - The extracted information can vary based on the configure, but will always include the tracks.
    - Contact someone at Berkeley to see if the data is already skimmed. I'd suggest starting with Tucker, but if he's not available, you can follow up with Peter to find the right person.
  - The Berkeley Trees can be directly read by mammoth. Due to the evolving nature of Run 3 data, the format occasionally changes, so small adaptations may be needed.
- Other sources: e.g. JETSCAPE, JEWEL, pythia
  - See the `src/mammoth/framework/io` module, which implements reading or generation for a number input formats. The stored files are read into the format expected in the mammoth analysis framework, and then it should just run
  - io specific customizations are possible. See the io module.
  - Generator specific customization is possible at the analysis level. For an example, see the [reclustered substructure analysis code](https://github.com/raymondEhlers/mammoth/blob/main/src/mammoth/reclustered_substructure/analyze_chunk.py), focusing on the functionality related to the `mammoth.framework.analysis.generator_settings` module.

## Shared arguments

There are many shared settings across multiple runs of the same analysis, or across analyses over the same collision system (e.g. PbPb_MC).
To support these cases, we define a number of keys that store commonly used arguments, and use YAML anchors to make them easy to insert into different analyses.

> [!warning]
> These are **generally not** inserted automatically - it's up to you use a user to insert them into a configuration.

### Generator arguments

Stored under the `generator_arguments` key, these are collision system specific default arguments for handling generators.
Most are fairly minimal, but the JEWEL case shows what more extensive examples can do.

### Dataset metadata settings

These are collision system specific settings. Most are about mapping into expected framework conventions, but the embedding is a bit more complicated. In that case, there's a lot that can be configured, but the defaults are generally fine.

## General analysis parameters

You'll likely run an analysis multiple times, so it's convenient to share the settings over multiple productions. These are stored under the `analysis_settings` key, with a section for each analysis. Below, we'll use the settings for the entropy-based correlators as an example:

```yaml
analysis_settings:
  _EBC_analysis_settings:
    # Often settings are shared - even across collision systems - so it's useful to default a default settings
    default_parameters:
      # Control how outputs are stored
      default_output_settings:
        &EBC_output_settings # We look for the primary output to judge if we've already run the analysis for a given input file
        primary_output:
          # Mark either skim or histogram outputs as as the primary output
          type: "skim"
          # Settings for accessing the primary output
          reference_name: "tree"
          container_name: ""
        # Skim related
        # Should we return the skim to the analysis framework for further analysis beyond writing it?
        return_skim: false
        # Should we write the skim for each chunk? This is often the simplest approach
        write_chunk_skim: true
        # How should we write the skim? Options are "parquet" and "root"
        write_skim_extension: "parquet"
        # A technical settings for how parquet handles structures in the skim. Leave false unless writing fails.
        explode_skim_fields_to_separate_directories: false
        # Histogram related
        # Return merged histograms to the analysis framework
        return_merged_hists: false
        # Write a separate histogram per chunk
        write_chunk_hists: false
        # Write a single histogram merged over all the chunks for a single file
        write_merged_hists: true

    # pp related analysis setting
    pp: &analysis_settings_EBC_pp # Share the trigger settings
      trigger_parameters: &analysis_settings_EBC_dijet_trigger
        classes:
          leading: [10., 140.]
          subleading: [10., 140.]
        parameters:
          jet_R: 0.4
      # What level are we analyzing?
      analysis_input_levels: ["data"]
      # In the way that we've implemented it as of Oct 2025, this is the maximum delta
      # phi from the trigger recoil axis
      max_delta_phi_from_axis: 0.7853981633974483 # pi/4
      # Selecting the minimum track pt per analysis input level
      # In GeV
      min_track_pt:
        data: 0.15

    pp_MC: &analysis_settings_EBC_pp_MC
      trigger_parameters:
        <<: *analysis_settings_EBC_dijet_trigger
      analysis_input_levels: ["det_level", "part_level"]
      # In the way that we've implemented it as of Oct 2025, this is the maximum delta
      # phi from the trigger recoil axis
      max_delta_phi_from_axis: 0.7853981633974483 # pi/4
      # Selecting the minimum track pt per analysis input level
      # In GeV
      min_track_pt:
        part_level: 0.15
        det_level: 0.15
      # By default, don't apply any additional reduction.
      # This can be used to implement the tracking efficiency uncertainty systematic.
      # There is also a pt dependent version
      det_level_artificial_tracking_efficiency: 1.0
```

> [!tip]
> All of the high level keys (e.g. `pp_MC` inside of `_EBC_analysis_settings`) are defined by convention. You could name them anything - they're access through the YAML anchors.

## Analysis settings

.....

# Productions

> [!tip]
> I recommend storing the production.yaml files in a git repository. In conjunction with the track skim config, this will uniquely specify how each production was run.

- [ ] run_analysis.py
  - Use one of the existing tasks as an example
  - Note that some analysis (e.g. the time reclustering) use a notebook for executing the analysis
- [ ] What do I need to edit?

# Running the analysis

For general information on running the analysis code, see [docs/running_an_analysis.md].

The analysis is run via `run_analysis` module in this directory - i.e. `python3 -m mammoth.entanglement_entropy.run_analysis`.
You can configure the analysis:

- Running / execution options in the `main(...)` function
- Running with parsl or dask in the call to `main(...)`
- Configure the analysis parameters in the 'track skim config' yaml in the `src/mammoth/alice/config` directory

# Practical information

Below are a few tips and tracks for running analyses, based on experienced gathered using mammoth over the years.

## Defining cluster environments

Mammoth handles job submission through the parsl and dask job submission frameworks.
Most of this is taken care of for the user, so you can focus on defining your analysis.
However, if you're running a cluster where no one has run before, you'll be responsible for defining the "facility configuration".
This configuration, which is specified in the `src/mammoth/job_utils` module, specifies how the cluster is defined, which in tern determines how to define batch jobs. The `Facility` object should be added into the `_facilities_configs` dict in the module. An example configuration is shown below:

```python
_facilities_configs.update(
    {
        # I loop over multiple queues, which have the same settings, but different wall times.
        # This allows me to easily vary the queue at runtime.
        f"hiccup_staging_{queue}": Facility(
            # Pick a name for clarity
            name="hiccup_staging",
            # Cluster definition
            node_spec=NodeSpec(n_cores=20, memory=64),
            partition_name=queue,
            # Allocate full node at a time:
            # target_allocate_n_cores=20,
            # Allocate by core:
            target_allocate_n_cores=1,
            launcher=SingleNodeLauncher,
            # Location of the scratch directory on the worker node.
            # Some environment variables, such as $USER will be expanded on the node to allow you
            # to ensure that paths are unique.
            node_work_dir=Path("/scratch/u/$USER/parsl"),
            # Base permanent storage path on the cluster
            storage_work_dir=Path("/rstorage"),
            # Option to exclude nodes for whatever reason.
            nodes_to_exclude=[],
            # Minimize file IO using file staging and reducing IO dependent checks.
            minimize_IO_as_possible=True,
        )
        for queue in ["quick", "std", "long"]
    }
)
```

> [!warning]
> Although parsl and dask should be able to support other job submissions systems, I've only ever used or tested it with slurm.

> [!tip]
> For those in JETSCAPE who have used the stat-xsede framework, this is facility configuration is quite similar, although they have diverged at various points.

### File staging

It's good manners on clusters to stage files to local scratch storage for analysis, thereby reducing load on the storage system.
This will also usually improve performance.
Mammoth includes built-in support for doing this at the framework level - the user doesn't need to think about it.
All you need to do is to enable `minimize_IO_as_possible` in the `Facility` configuration, and it will be handled transparently to the user.

> [!tip]
> You can temporarily disable this setting if needed for e.g. testing by enabling `override_minimize_IO_as_possible` when calling `setup_job_framework`.

## Debug mode

While your full analysis will generally need to run on a cluster, it's often convenient to run a test analysis on your local computer - did you implement that cut correctly? Does it run as expected? etc.

- [ ] Something about debug mode

### Debugging scheduler problems

Since dask and parsl schedule on different notes, they can be difficult to debug at times, especially if the issue is in the scheduler or deep in the mammoth framework. Often times, it can give errors that are nonspecific or related to file syncing that just indicate that the job failed.

- [ ] A quick note on the local executor.

> [!tip]
> Parsl doesn't run on macOS, so this can also be extremely useful when testing on a macOS device.
