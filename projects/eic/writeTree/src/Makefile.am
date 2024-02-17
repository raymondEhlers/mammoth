AUTOMAKE_OPTIONS = foreign

AM_CPPFLAGS = \
  -I$(includedir) \
  -I$(OFFLINE_MAIN)/include \
  -I`root-config --incdir`

lib_LTLIBRARIES = \
   libWriteTree.la

libWriteTree_la_LDFLAGS = \
  -L$(libdir) \
  -L$(OFFLINE_MAIN)/lib

libWriteTree_la_LIBADD = \
  -lfun4all \
  -lphhepmc \
  -lphhepmc_io

libWriteTree_la_SOURCES = \
  writeTree.cc

pkginclude_HEADERS = \
  writeTree.h


################################################
# linking tests

noinst_PROGRAMS = testexternals

BUILT_SOURCES = \
  testexternals.cc

testexternals_SOURCES = \
  testexternals.cc

testexternals_LDADD = \
  libWriteTree.la

testexternals.cc:
	echo "//*** this is a generated file. Do not commit, do not edit" > $@
	echo "int main()" >> $@
	echo "{" >> $@
	echo "  return 0;" >> $@
	echo "}" >> $@

clean-local:
	rm -f $(BUILT_SOURCES)
