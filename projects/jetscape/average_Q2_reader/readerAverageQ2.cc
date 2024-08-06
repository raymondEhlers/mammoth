/*******************************************************************************
 * Copyright (c) The JETSCAPE Collaboration, 2018
 *
 * Modular, task-based framework for simulating all aspects of heavy-ion
 *collisions
 *
 * For the list of contributors see AUTHORS.
 *
 * Report issues at https://github.com/JETSCAPE/JETSCAPE/issues
 *
 * or via email to bugs.jetscape@gmail.com
 *
 * Distributed under the GNU General Public License 3.0 (GPLv3 or later).
 * See COPYING for details.
 ******************************************************************************/
// Reader test (focus on graph)

#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <thread>

#include "JetScapeBanner.h"
#include "JetScapeLogger.h"
#include "JetScapeReader.h"
#include "PartonShower.h"
#include "fjcore.hh"
#include "gzstream.h"

#include <GTL/dfs.h>

using namespace std;
// using namespace fjcore;

using namespace Jetscape;

// -------------------------------------

// Forward declaration
void Show();
void AnalyzeGraph(shared_ptr<PartonShower> mS);
void LeadingProng(shared_ptr<PartonShower> mS);
ostream &operator<<(ostream &ostr, const fjcore::PseudoJet &jet);

// -------------------------------------

// Create a pdf of the shower graph:
// Use with graphviz (on Mac: brew install graphviz --with-app)
// in shell: dot GVfile.gv -Tpdf -o outputPDF.pdf
// [or you can also use the GraphViz app for Mac Os X (in the "cellar" of
// homebrew)]

// -------------------------------------

int main(int argc, char **argv) {
  JetScapeLogger::Instance()->SetDebug(false);
  JetScapeLogger::Instance()->SetRemark(false);
  // SetVerboseLevel (9 a lot of additional debug output ...)
  // If you want to suppress it: use SetVerboseLevle(0) or max
  // SetVerboseLevle(9) or 10
  JetScapeLogger::Instance()->SetVerboseLevel(0);

  cout << endl;
  Show();

  // Do some dummy jetfinding ...
  fjcore::JetDefinition jet_def(fjcore::antikt_algorithm, 0.7);

  vector<shared_ptr<PartonShower>> mShowers;

  // Directly with template: provide the relevant stream
  // auto reader=make_shared<JetScapeReader<ifstream>>("test_out.dat");
  // auto reader=make_shared<JetScapeReader<igzstream>>("test_out.dat.gz");

  // Hide Template (see class declarations in reader/JetScapeReader.h) ...
  auto reader = make_shared<JetScapeReaderAscii>("test_out.dat");
  // auto reader=make_shared<JetScapeReaderAsciiGZ>("test_out.dat.gz");

  // reads in multiple events and multiple shower per event
  // commented out so that you get the dot graph file for the first shower in
  // the first event (add in and the file gets overridden)
  int iEvent = 0;
  //while (!reader->Finished())
  {
    reader->Next();

    cout << "Analyze current event = " << reader->GetCurrentEvent() << endl;
    mShowers = reader->GetPartonShowers();

    // Find leading parton shower
    std::size_t leadingShowerIndex = 0;
    double leadingShowerPt = 0;
    for (std::size_t i = 0; i < mShowers.size(); i++) {
      cout << " Analyze parton shower = " << i << ", leading shower index=" << leadingShowerIndex << std::endl;

      mShowers[i]->PrintVertices();
      mShowers[i]->PrintPartons();

      auto partonPseudoJet = mShowers[i]->GetPartonAt(0)->GetPseudoJet();
      if (partonPseudoJet.pt() > leadingShowerPt) {
        leadingShowerIndex = i;
        leadingShowerPt = partonPseudoJet.pt();
      }
    }

    // AnalyzeGraph(mShowers[i]);
    LeadingProng(mShowers[leadingShowerIndex]);

    if (iEvent == 0) {
      mShowers[leadingShowerIndex]->SaveAsGV("my_test.gv");
      mShowers[leadingShowerIndex]->SaveAsGML("my_test.gml");
      mShowers[leadingShowerIndex]->SaveAsGraphML("my_test.graphml");
    }

    /*
    cout << " Found " << finals << " final state partons." << endl;
    auto hadrons = reader->GetHadrons();
    cout<<"Number of hadrons is: " << hadrons.size() << endl;

    fjcore::ClusterSequence hcs(reader->GetHadronsForFastJet(), jet_def);
    vector<fjcore::PseudoJet> hjets =
    fjcore::sorted_by_pt(hcs.inclusive_jets(2)); cout<<"AT HADRONIC LEVEL " <<
    endl; for (int k=0;k<hjets.size();k++) cout<<"Anti-kT jet "<<k<<" :
    "<<hjets[k]<<endl;
    */

    // for(unsigned int i=0; i<hadrons.size(); i++) {
    // 	cout<<"For Hadron Number "<<i<<" "<< hadrons[i].get()->e() << " "<<
    // hadrons[i].get()->px()<< " "<< hadrons[i].get()->py() << " "<<
    // hadrons[i].get()->pz()<< " "<< hadrons[i].get()->pt()<<  endl;
    // }
    iEvent += 1;
  }

  reader->Close();
}

// -------------------------------------

void AnalyzeGraph(shared_ptr<PartonShower> mS) {
  JSINFO << "Some GTL graph/shower analysis/dfs search output:";

  // quick and dirty ...
  dfs search;
  search.calc_comp_num(true);
  search.scan_whole_graph(true);
  search.start_node(); // defaulted to first node ...
  search.run(*mS);

  cout << endl;
  cout << "DFS graph search feature from GTL:" << endl;
  cout << "Number of Nodes reached from node 0 = "
       << search.number_of_reached_nodes() << endl;
  cout << "Node/Vertex ordering result from DFS:" << endl;
  dfs::dfs_iterator itt2, endt2;
  for (itt2 = search.begin(), endt2 = search.end(); itt2 != endt2; ++itt2) {
    cout << *itt2 << " "; //<<"="<<search.dfs_num(*itt2)<<" ";
  }
  cout << endl;
  cout << "Edge/Parton ordering result from DFS:" << endl;
  dfs::tree_edges_iterator itt, endt;
  for (itt = search.tree_edges_begin(), endt = search.tree_edges_end();
       itt != endt; ++itt) {
    cout << *itt; //<<endl;
  }
  cout << endl;

  dfs::roots_iterator itt3, endt3;
  cout << "List of root nodes found in graph/shower : ";
  for (itt3 = search.roots_begin(), endt3 = search.roots_end(); itt3 != endt3;
       ++itt3) {
    cout << **itt3;
  }
  cout << endl;
  cout << endl;
}

bool IsEndNode(node n) {
  if (n.indeg() > 0 && n.outdeg() == 0)
    return true;
  else
    return false;
}

bool IsOneToTwo(node n) {
  if (n.indeg() == 1 && n.outdeg() == 2)
    return true;
  else
    return false;
}

edge GetHighSplitEdge(shared_ptr<PartonShower> mS, node n) {
  if (n.outdeg() < 1) {
    throw std::logic_error("Exhausted partons in shower.");
  }

  // Find the leading parton and kinematics
  auto leadingEdge = *n.out_edges_begin();
  auto leadingPseudoJet = mS->GetParton(leadingEdge)->GetPseudoJet();
  for (auto edgeIter = n.out_edges_begin(); edgeIter != n.out_edges_end(); ++edgeIter) {
    // Find the leading parton in the outgoing edges
    fjcore::PseudoJet currentPseudoJet = mS->GetParton(*edgeIter)->GetPseudoJet();
    //std::cout << "currentPseudoJet: " << currentPseudoJet << std::endl;

    // Store the new leading parton edge
    if (currentPseudoJet.pt() > leadingPseudoJet.pt()) {
      leadingEdge = *edgeIter;
      leadingPseudoJet = currentPseudoJet;
    }
  }
  return leadingEdge;
}

void LeadingProng(shared_ptr<PartonShower> mS) {
  cout << endl;
  cout << "Get leading prong in shower graph:" << endl;
  cout << endl;
  node n = mS->GetNodeAt(1);
  fjcore::PseudoJet pIn = mS->GetPartonAt(0)->GetPseudoJet();

  cout << n << " " << IsOneToTwo(n) << " " << pIn << endl;

  try {
    while (!IsEndNode(n)) {
      edge e = GetHighSplitEdge(mS, n);
      auto parton = mS->GetParton(e);
      fjcore::PseudoJet p = parton->GetPseudoJet();
      n = e.target();

      cout << std::boolalpha << n << " " << IsOneToTwo(n) << " " << p
           << ", time=" << parton->time() << ", virtuality=" << parton->t() << endl;
    }
  } catch (std::logic_error &e) {
    // Do nothing...
  }
}

// -------------------------------------

void Show() {
  ShowJetscapeBanner();
  INFO_NICE;
  INFO_NICE << "------------------------------------";
  INFO_NICE << "| Reader Test JetScape Framework ... |";
  INFO_NICE << "------------------------------------";
  INFO_NICE;
}

//----------------------------------------------------------------------
/// overloaded jet info output

ostream &operator<<(ostream &ostr, const fjcore::PseudoJet &jet) {
  if (jet == 0) {
    ostr << " 0 ";
  } else {
    ostr << " pt = " << jet.pt() << " m = " << jet.m() << " y = " << jet.rap()
         << " phi = " << jet.phi();
  }
  return ostr;
}

//----------------------------------------------------------------------
