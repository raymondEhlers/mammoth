# Running an analysis

After [developing an analysis](developing_an_analysis.md), you need to follow some conventions to successfully take advantage of the analysis framework[^1]. Below, we'll use an example analysis to guide through this process.

[^1]: For clarity, if you have some separately developed piece of code, you don't necessarily need to run it through the analysis framework. The framework is designed to help you run your analysis (particularly in parallel), but you can also go it alone. If you decide to go that route, I'd suggest that you clearly label that code as such.

> [!tip]
> I've tried to document the main concepts below, but the bottom line is this documentation cannot cover everything at this time. If you're trying to do something, I recommend searching around the repository. What you're looking for has probably already been done for some analysis, and so you can adapt that existing example for your own purposes.

# Key concepts

Each run of an analysis is known as a "production". This corresponds to a single run of a given analysis code over a single dataset with a specified set of analysis parameters. For example, this could be analysis of groomed jet Rg over the ALICE 2024 pp reference dataset for R = 0.4 jets, soft drop z_cut = 0.2, etc... ([We cover a particular production dataset and analysis parameters below](#defining-and-running-a-production), but for now, let's stick to higher level parameters). Each production is defined for a collision system, and is assigned a unique number within that system.

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

# Defining and running a production

Finally, a production number and collision system are created for an analysis.

> [!note]
> By convention, productions that we want to store begin from number 60. Numbers below that are considered test productions, and are transient.

Consider the example for an entropy-based correlator analysis in pp collisions using Run 3 data. We'll use comments in this example to describe the how to define these arguments.

```yaml
productions:
  pp:
    64:
      # EBC with Run 3 pp_ref w/ some bug fixes
      # (n.b. It's helpful to include a brief summary comment, just to help keep track of what is what)
      metadata:
        # Start with the base pp metadata defined further up in the config
        <<: *base_metadata_pp
        # And we're going to analyze the pp_ref pass1 dataset defined further up in the config
        dataset: *LHC24_pp_ref_pass1
        # This production number should match the number used as the key (i.e. the key below 'pp')
        production_number: 64
        # You can add an additional label to the outputs. e.g. if you're analyzing central and semi_central separately.
        # Here, we don't need it, so we set it to be empty
        label: ""
      settings:
        # n.b. These settings **MUST** align with the argument names defined in analyze_chunk for the given analysis.
        #      There are some general parameters, and then analysis specific parameters.
        # Use the base EBC pp settings that we defined above
        <<: *analysis_settings_EBC_pp
        # How many events are included in a chunk that we analyze.
        chunk_size: 10000
        # How many of those chunks do we analyze before stop using an input file?
        # For run 2 analyses, this is usually omitted since the files are much more split up.
        # For run 3 analyses, the files are much larger, so it can be useful if you're trying to run a quicker test.
        # NOTE: These are general statements, but they depend on how the track skim was constructed.
        # NOTE: When you want the full statistics, you **MUST** remove this max value and let it run
        #       over the input file until it's exhausted.
        max_n_chunks: 50
        # Analysis output settings that we defined above.
        output_settings:
          <<: *EBC_output_settings
        # Here, we just take the default analysis parameters, but we could override any of the default analysis settings here.
        # e.g. we could set:
        #max_delta_phi_from_axis: 0.5
```

The production settings include the dataset settings and the analysis settings. The dataset settings define the IO, while the analysis settings match

> [!tip]
> The analysis settings **MUST** align with the argument names defined in analyze_chunk for the given analysis.
> There are some general parameters, and then analysis specific parameters.

> [!tip]
> There are many more possible arguments than I show here. Check out some other analyses to see what else is possible.

## Executing the analysis

An analysis should be run via a `run_analysis` or `steering` module in the project src directory - e.g. `python3 -m mammoth.entanglement_entropy.run_analysis`.
The name is set by convention, but is not strictly required.
This module will usually contain:

- Your ProductionSpecialization class ([see developing an analysis](developing_an_analysis.md)).
- Setting up your workflow apps via `setup_framework_default_workflows` ([see developing an analysis](developing_an_analysis.md)).
- A function to define the Productions(s) you want to run:

```python
def define_productions() -> list[production.ProductionSettings]:
    # We want to provide the opportunity to run multiple productions at once.
    # We'll do so by defining each production below and then iterating over them below
    productions = []

    # Create and store production information
    _here = Path(__file__).parent
    # This assumes that this file is stored in the `src/mammoth/<analysis>/` directory.
    config_filename = Path(_here.parent / "alice" / "config" / "track_skim_config.yaml")
    productions.extend(
        # This is a list, so you could submit multiple productions at once.
        [
            production.ProductionSettings.read_config(
                collision_system="pp",
                number=64,
                # Specialization for the entropy-based correlators analysis
                specialization=EBCProductionSpecialization(),
                # Location of the track skim config file.
                track_skim_config_filename=config_filename,
                # Inform the framework the base directory where the input track skim files are stored
                base_output_dir=Path("/rstorage/rehlers/trains"),
            ),
        ]
    )

    # Write out the production settings
    for production_settings in productions:
        production_settings.store_production_parameters()

    return productions
```

- A function to setup and submit tasks, usually called `setup_and_submit_tasks(...)`. Function names called there are set by convention, so you can usually blindly copy this function from any other analysis.
- The `run(...)` function that is called to execute the analysis:

```python
def run(job_framework: job_utils.JobFramework) -> list[Future[Any]]:
    # Job execution parameters
    productions = define_productions()
    task_name = "EBC_mammoth"

    # Job execution configuration
    # Can be skipped unless ROOT is needed (and then you need to setup a consistent conda environment)
    conda_environment_name = ""
    # Define the resources needed for a single job
    task_config = job_utils.TaskConfig(name=task_name, n_cores_per_task=1)
    # How many tasks we should aim to run at once
    target_n_tasks_to_run_simultaneously = 110
    # Logging
    log_level = logging.INFO
    # Job walltime
    walltime = "24:00:00"
    # Whether you want to override the IO minimization setting
    override_minimize_IO_as_possible = None
    # Debug mode is covered below. It's useful for testing and fixing issues, but SHOULD NOT be used for a full production run.
    debug_mode = False
    if debug_mode:
        # Usually, we want to run in the short queue so that debugging runs faster
        target_n_tasks_to_run_simultaneously = 2
        walltime = "1:59:00"
    # This uses the "hiccup" cluster at LBL. You can configure this with a simple string, but it's often nice to adapt
    # based on the walltime.
    # NOTE: This name goes back to the facility names that we defined in job_utils.
    facility: job_utils.FACILITIES = (
        "hiccup_staging_std"
        if job_utils.hours_in_walltime(walltime) >= 2
        else "hiccup_staging_quick"
    )

    # Setup the job framework that we've selected based on your settings.
    # Keep the job executor just to keep it alive
    job_executor, _job_framework_config, execution_settings = setup_job_framework(
        job_framework=job_framework,
        productions=productions,
        task_config=task_config,
        facility=facility,
        walltime=walltime,
        target_n_tasks_to_run_simultaneously=target_n_tasks_to_run_simultaneously,
        log_level=log_level,
        conda_environment_name=conda_environment_name,
        override_minimize_IO_as_possible=override_minimize_IO_as_possible,
        debug_mode=debug_mode,
    )
    # Submits all of the jobs over all of the input files, as defined in the production configuration.
    all_results = setup_and_submit_tasks(
        productions=productions,
        task_config=task_config,
        execution_settings=execution_settings,
        job_executor=job_executor,
    )

    # Keeps track of the "futures" that correspond to eventual outputs from the tasks.
    # This will drop you into an IPython shell, where you can check the status of the futures
    # stored in `all_results`. You can close the shell with ctrl-d, or via `exit()`.
    # If all of your futures have not yet resolved when it close it, it will wait until they're all done.
    process_futures(
        productions=productions, all_results=all_results, job_framework=job_framework
    )

    return all_results


# This ensures that the run(...) function is executed with it is called with `python3 -m path.to.module` ...
if __name__ == "__main__":
    # This sets the job running framework. Possible options are:
    # job_utils.JobFramework.parsl
    # job_utils.JobFramework.dask_delayed
    # job_utils.JobFramework.immediate_execution_debug
    # Some thoughts on these options are listed below
    run(job_framework=job_utils.JobFramework.dask_delayed)
```

> [!tip]
> Go check out `src/mammoth/entanglement_entropy/run_analysis.py` for a complete example. This can provide a better overview than is possible by just looking at snippets.

> [!tip]
> You can also execute the steering through a jupyter notebook rather than calling the module through python. See `projects/time_reclustering/steering.py` (convert via jupytext). This can be particularly convenient if you want to do downstream steps, such as unfolding.

# Practical information

Below are a few tips and tricks for running analyses, based on experienced gathered using mammoth over the years.

## Tips for running?

At this point, a reasonable question is to ask: What do I usually need to edit?
Most of the time, it's pretty simple:

- Configure the production and analysis parameters in the 'track_skim_config.yaml' in the `src/mammoth/alice/config` directory
- Modify the production settings in the `define_productions(...)` to execute that particular production.
- Modify the execution options in the `main(...)` function, as needed (often, few changes are needed once you've figured out how to run an analysis on a cluster)
- Decide which job framework to use in the call to `main(...)` (see the practical information below).

> [!tip]
> When you run an analysis, it will store a `production.yaml` file that includes the packages used to run the analysis, the analysis arguments, etc. I recommend storing this file in a git repository. In conjunction with the track skim config and the repo state, this uniquely specify how each production was run!

## Choosing a job framework

Parsl and dask distributed are both available for scheduling jobs with e.g. slurm.
Both do approximately the same thing, but each has its quirks.
To briefly summarize the two options:

- [parsl](https://parsl-project.org/) was developed in the academic HPC world. It has example configurations for many US HPC systems.
- [dask distributed](ihttps://distributed.dask.org/en/latest/) was developed as part of the broader dask project, which handles all kind of distributed computing.

Both work, so pick your favorite.
I usually pick the one that I'm most comfortable / have the most experience with on a given cluster.
I would suggest that you arbitrarily pick one, go read a little bit about it, and if it sounds good, stick with it.
If you have an issue, try switching to the other one.

> [!tip]
> Recall that since these frameworks handle job scheduling with slurm, the process need to stay alive while jobs are running. If you run as recommended above, it should do that by default.

### Debugging scheduler problems

Since dask and parsl schedule on different cluster nodes, they can be difficult to debug at times, especially if the issue is in the scheduler or deep in the mammoth framework.
Often times, it can give errors that are nonspecific or related to file syncing that just indicate that the job failed.
To work around this, there is a local, immediate execution job framework that can be used for debugging purposes.
You can use it by setting the `job_framework` argument to `run(...)` to `job_utils.JobFramework.immediate_execution_debug`.

> [!tip]
> Parsl doesn't run on macOS (known issue, unlikely to be fixed soon), so this can also be extremely useful when testing on a macOS device.

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
To help answer these questions, there's a debug mode, which you can set via the `debug_mode` argument to `setup_job_framework`.
You can set it in a few different ways:

- `debug_mode=True`: This will enable debug mode, and start processing the first 2 files.
- `debug_mode={<some files>}`. For example, `debug_mode: dict[str | int, Any] = {20: [Path("trains/pythia/2640/run_by_run/LHC20g4/296244/20/AnalysisResults.20g4.011.root")]}`, which will use the file `AnalysisResults.20g4.011.root` with pt hat bin 20.
  - You can pass multiple files, etc.
  - **YOU** are responsible for passing files in the right format for the collision system that you're running. What you need to pass depends on whether data vs MC, regular vs embedding, etc. You can check out the `mammoth.framework.steer_workflow` module to see some examples of what those should look like.

> [!tip]
> I highly recommend running this in conjunction with the `immediate_execution_debug` job framework. Otherwise, it can very difficult to reason about multiple processes running in parallel. Plus, the logging is worse when running on different nodes.
