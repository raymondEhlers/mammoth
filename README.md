# Mammoth

## Install for other projects

First, remove `pachyderm` from the dependencies, because it probably won't be picked up correctly by pip.
Next, run

```bash
# Actually build the extensions...
$ pip install --use-feature=in-tree-build ../mammoth

# We've built in the tree, so now we need to do an editable install so it can find the extensions...
$ pip install -e ../mammoth
```
