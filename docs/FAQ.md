# Running the framework

## Some of my jobs failed due to losing the manager - can I recover the outputs??

If it's running under parsl, then staging out files may have failed when the manager was lost. In that case, we can recover the files on each node by logging in to each node and running e.g.:

```zsh
# Load the mammoth virtulanev. e.g.:
$ source /software/users/rehlers/dev/mammoth/.venv-3.11/bin/activate
# And then use the script to handle failed outputs
$ handle_files_from_failed_jobs -n /scratch/u/rehlers/parsl -p /rstorage
```

Note that `/rstorage` is the base path in the cluster config in `job_utils` - eg. it's `/rstorage` on hiccup, even though the `base_output_dir` from the production may be `/rstorage/rehlers/trains`. You can check this by going to one of the output directories on the node. The path that you get from e.g. `/scratch/u/rehlers/parsl/<UUID>/output/rehlers/...` means that you want the parent of `rehlers` for the permanent storage dir. In this case, that would be `/rstorage`.

If this is really a big issue, one could imagine making a quick slurm script that only runs one job per node. However, for something the size of hiccup, it's probably fine to just login to each system via ssh and run by hand.

## Some of my jobs failed while using staging - do I need to anything?

It's the same sort of scenario as [Some of my jobs failed due to losing the manager - can I recover the outputs??](#some-of-my-jobs-failed-due-to-losing-the-manager---can-i-recover-the-outputs) -- see above.

## What do I do about random job files that are still leftover in scratch?

If for some reason the `handle_...` script doesn't fully cleanup the scratch files, there's an additional script to handle additional cleanup: `cleanup_files_from_failed_jobs`.
This could happen for e.g. jobs that field long ago or didn't otherwise have salvageable outputs.
Of particular note here is that it handles `ssh` login to the various systems, assuming you have passwordless authentication setup.
See the options with `cleanup_files_from_failed_jobs --help`.

> [!warning]
> Use with care and prefer using `handle_files_...` as much as possible. This script deletes fairly aggressively, so it requires explicit user confirmation for each deletion.

> [!tip]
> Due to the risk of this script, there are a number of hostname checks, etc. You may need to make some small edits to the script to use it on a new system. Use with care!
