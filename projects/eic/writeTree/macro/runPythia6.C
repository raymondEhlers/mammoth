#ifndef MACRO_FUN4ALLJETANA_C
#define MACRO_FUN4ALLJETANA_C

#include <fun4all/Fun4AllDstInputManager.h>
#include <fun4all/Fun4AllInputManager.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/SubsysReco.h>

//#include <PHPythia6.h>
//#include <PHPy6GenTrigger.h>
//#include <PHPy6JetTrigger.h>
// Custom copy. All must come from the same place...
R__ADD_INCLUDE_PATH("/software/rehlers/dev/eic/install/include")
#include <phpythia6/PHPythia6.h>
#include <phpythia6/PHPy6GenTrigger.h>
#include <phpythia6/PHPy6JetTrigger.h>

#include <writetree/writeTree.h>

// here you need your package name (set in configure.ac)
R__LOAD_LIBRARY(libfun4all)
R__LOAD_LIBRARY(libPHPythia6)
R__LOAD_LIBRARY(libWriteTree)

void Simple(const int nevent = 1e4, const int index = 0, const double minP = 100, const std::string outputPath = "/alf/data/rehlers/eic/pythia6/sim")
{
  Fun4AllServer *se = Fun4AllServer::instance();
  //se->Verbosity(4);

  PHPythia6 * pythia6 = new PHPythia6();

  // Configure
  std::stringstream outputFile;
  pythia6->set_config_file("/software/rehlers/dev/mammoth/projects/eic/writeTree/phpythia6_ep.cfg");
  // Write HepMC
  //pythia6->save_ascii("test.hepmc");

  PHPy6JetTrigger *trig = new PHPy6JetTrigger();
  trig->SetEtaHighLow(1, 4);
  trig->SetMinJetP(minP);
  //trig->SetMinJetPt(5);
  trig->SetJetR(0.7);
  //trig->Verbosity(4);
  pythia6->register_trigger(trig);
  //pythia6->Verbosity(4);
  se->registerSubsystem(pythia6);

  outputFile.str(outputPath);
  outputFile << "/writeTree_nevents_";
  outputFile << nevents;
  outputFile << "_p_trigger_";
  outputFile << static_cast<int>(minP);
  outputFile << ".root";
  WriteTree *writeTree = new WriteTree(outputFile.str());
  se->registerSubsystem(writeTree);

  /*MyJetAnalysis *myJetAnalysis = new MyJetAnalysis("AntiKt_Tower_r04", "AntiKt_Truth_r04", "myjetanalysis.root");
  //  myJetAnalysis->Verbosity(0);
  // change lower pt and eta cut to make them visible using the example
  //  pythia8 file
  myJetAnalysis->setPtRange(1, 100);
  myJetAnalysis->setEtaRange(-1.1, 1.1);
  se->registerSubsystem(myJetAnalysis);

  Fun4AllInputManager *in = new Fun4AllDstInputManager("DSTin");
  in->fileopen(inputfile);
  se->registerInputManager(in);*/

  se->run(nevent);
  se->End();
  delete se;
  gSystem->Exit(0);
}

#endif
