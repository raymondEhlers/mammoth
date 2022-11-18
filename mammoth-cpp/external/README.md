# fjcontrib

## rpath support

`fjcontrib` doesn't support passing LDFLAGS into the Makefile - they're just ignored.
So we created a patch to add this support. It needs to be applied before configuring fjcontrib.
It was created with:

```bash
$ diff -Naur fastjet/fjcontrib-1.048/Makefile.in Makefile.in > fjcontrib_ldflags_rpath.patch
```

where the `Makefile.in` in the local directory contains the changes. The patch can be applied with:

```bash
$ cd fastjet/fjcontrib-1.048
$ patch < ../../fjcontrib_ldflags_rpath.patch
```
