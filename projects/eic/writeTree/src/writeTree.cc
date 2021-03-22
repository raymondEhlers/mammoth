#include "writeTree.h"

#include <fun4all/Fun4AllReturnCodes.h>
#include <fun4all/Fun4AllServer.h>
#include <fun4all/PHTFileServer.h>

#include <phool/PHCompositeNode.h>
#include <phool/getClass.h>

#include <phhepmc/PHHepMCGenEvent.h>
#include <phhepmc/PHHepMCGenEventMap.h>

#include <HepMC/GenEvent.h>

#include <TFile.h>
#include <TString.h>
#include <TTree.h>
#include <TVector3.h>

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>

using namespace std;

template<typename T, typename... Args>
std::unique_ptr<T> my_make_unique(Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

WriteTree::WriteTree(const std::string& outputfilename)
  : SubsysReco("WriteTree")
  , m_outputFileName(outputfilename)
  , m_T(nullptr)
  , m_event(-1)
  , m_x1(-1)
  , m_x2(-1)
  , m_q2(-1)
  , m_particleID{}
  , m_particleStatus{}
  , m_pt{}
  , m_eta{}
  , m_phi{}
  , m_E{}
{
}

WriteTree::~WriteTree()
{
}

int WriteTree::Init(PHCompositeNode* topNode)
{
  if (Verbosity() >= WriteTree::VERBOSITY_SOME) {
    cout << "WriteTree::Init - Outoput to " << m_outputFileName << endl;
  }

  PHTFileServer::get().open(m_outputFileName, "RECREATE");

  //Trees
  m_T = new TTree("tree", "tree");

  m_T->Branch("event_ID", &m_event, "event_ID/I");
  m_T->Branch("x1", &m_x1, "x1/F");
  m_T->Branch("x2", &m_x2, "x2/F");
  m_T->Branch("q2", &m_q2, "q2/F");
  m_T->Branch("particle_ID", &m_particleID);
  m_T->Branch("status", &m_particleStatus);
  m_T->Branch("pt", &m_pt);
  m_T->Branch("eta", &m_eta);
  m_T->Branch("phi", &m_phi);
  m_T->Branch("E", &m_E);

  return Fun4AllReturnCodes::EVENT_OK;
}

int WriteTree::End(PHCompositeNode* topNode)
{
  cout << "WriteTree::End - Output to " << m_outputFileName << endl;
  PHTFileServer::get().cd(m_outputFileName);

  m_T->Write();

  return Fun4AllReturnCodes::EVENT_OK;
}

int WriteTree::InitRun(PHCompositeNode* topNode)
{
  return Fun4AllReturnCodes::EVENT_OK;
}

int WriteTree::process_event(PHCompositeNode* topNode)
{
  //if (Verbosity() >= WriteTree::VERBOSITY_SOME) {
  if (Verbosity() >= 3) {
      std::cout << "WriteTree::process_event() entered" << endl;
  }
  if (Verbosity() >= 2) {
      if (m_event % 1 == 0) {
          std::cout << "m_event: " << m_event << "\n";
      }
  }
  else {
      if (m_event % 100 == 0) {
          std::cout << "m_event: " << m_event << "\n";
      }
  }

  // Reset tree branches
  m_x1 = 0;
  m_x2 = 0;
  m_q2 = 0;
  m_particleID.clear();
  m_particleStatus.clear();
  m_pt.clear();
  m_eta.clear();
  m_phi.clear();
  m_E.clear();

  // Retrieve HepMC event generated in Pythia.
  auto phEventMap = findNode::getClass<PHHepMCGenEventMap>(topNode, "PHHepMCGenEventMap");
  if (phEventMap->size() > 1) {
      throw std::runtime_error("PHEventMap has more than one entry. This probably isn't handled right.");
  }
  //std::cout << "size: phEvent->size(): " << phEventMap->size() << "\n";

  HepMC::GenEvent * event = nullptr;
  for (auto ev = phEventMap->begin(); ev != phEventMap->end(); ++ev) {
      event = ev->second->getEvent();
      break;
  }
  //std::cout << event << "\n";

  // Keep track of event number. May not be necessary, but it's doesn't really hurt.
  ++m_event;

  // Event level info
  auto pdfInfo = event->pdf_info();
  m_x1 = pdfInfo->x1();
  m_x2 = pdfInfo->x2();
  m_q2 = pdfInfo->scalePDF();
  //unsigned int outgoingPartonPDGID = pdfInfo->id2();

  int i = 0;
  for (auto p = event->particles_begin(); p != event->particles_end(); ++p) {
      //std::cout << i << ": " << (*p)->pdg_id() << ", status: " << (*p)->status() << ", pz: " << (*p)->momentum().pz() << "\n";
      m_particleID.emplace_back((*p)->pdg_id());
      m_particleStatus.emplace_back((*p)->status());
      m_pt.emplace_back((*p)->momentum().perp());
      m_eta.emplace_back((*p)->momentum().eta());
      m_phi.emplace_back((*p)->momentum().phi());
      m_E.emplace_back((*p)->momentum().e());

      i++;
  }

  m_T->Fill();

  /*
  // Try to get some info myself...
  // NOTE: Careful: This is pythia e-p config specific, etc. May need to adjust
  //       if the config is modified.
  auto res = event->beam_particles();
  std::cout << "First beam particle pdg id: " << res.first->pdg_id() << "\n";
  std::cout << "Second beam particle pdg id: " << res.second->pdg_id() << "\n";
  //auto vertex = res.second->end_vertex();
  auto vertex = res.first->end_vertex();
  // NOTE: descendants vs children is the critical difference here.
  //for (HepMC::GenVertex::particle_iterator des = vertex->particles_begin(HepMC::descendants);
  //     des != vertex->particles_end(HepMC::descendants);
  //     ++des)
  for (HepMC::GenVertex::particle_iterator des = vertex->particles_begin(HepMC::children);
       des != vertex->particles_end(HepMC::children);
       ++des)
  {
      std::cout << "Descendent pdgid: " << (*des)->pdg_id() << ", status: " << (*des)->status() << "\n";
      //for (HepMC::GenVertex::particle_iterator des2 = (*des)->end_vertex()->particles_begin(HepMC::descendants);
      //     des2 != (*des)->end_vertex()->particles_end(HepMC::descendants);
      //     ++des2)
      //{
      //    std::cout << "\t->Descendent pdgid: " << (*des2)->pdg_id() << ", status: " << (*des2)->status() << "\n";
      //}
  }
  std::cout << "Done\n";

  // Proton beam particle.
  std::unique_ptr<HepMC::FourVector> incomingProton = nullptr;
  // Electron coming into the 2->2 scattering (post ISR)
  std::unique_ptr<HepMC::FourVector> incomingElectron = nullptr;
  // Electron coming out of the 2->2 scattering.
  std::unique_ptr<HepMC::FourVector> outgoingElectron = nullptr;
  unsigned int index = 0;
  for (auto p = event->particles_begin(); p != event->particles_end(); ++p) {
      std::cout << index << ": " << (*p)->pdg_id() << ", status: " << (*p)->status() << ", pz: " << (*p)->momentum().pz() << "\n";
      // This is pythia record 2
      if (index == 1) {
          std::cout << "Proton\n";
          if ((*p)->pdg_id() != 2212) { std::cout << "Not a proton in the proton beamline spot. Quitting!\n"; std::exit(1); }
          incomingProton = my_make_unique<HepMC::FourVector>((*p)->momentum());
      }
      // This is pythia record 5
      if (index == 4) {
          std::cout << "Incoming electron\n";
          if ((*p)->pdg_id() != 11) { std::cout << "Not an electron in the incoming electron spot. Quitting!\n"; std::exit(1); }
          incomingElectron = my_make_unique<HepMC::FourVector>((*p)->momentum());
      }
      // This is pythia record 7
      if (index == 6) {
          std::cout << "Outgoing electron\n";
          if ((*p)->pdg_id() != 11) { std::cout << "Not an electron in the outgoing electron spot. Quitting!\n"; std::exit(1); }
          outgoingElectron = my_make_unique<HepMC::FourVector>((*p)->momentum());
      }
      index++;
  }

  // Virtual photon corresponds to the difference in the four vectors.
  // Apparently they don't implement operators...
  HepMC::FourVector photon(incomingElectron->px() - outgoingElectron->px(),
                           incomingElectron->py() - outgoingElectron->py(),
                           incomingElectron->pz() - outgoingElectron->pz(),
                           incomingElectron->e() - outgoingElectron->e());
  std::cout << "Photon: px=" << photon.px() << ", py=" << photon.py() << ", pz=" << photon.pz() << ", e=" << photon.e() << "\n";

  double Q2 = - photon.m2();
  // Apparently they don't implement dot product...
  // Using + - - - signature.
  double dotProduct = (incomingProton->e() * photon.e()) - (incomingProton->px() * photon.px()) - (incomingProton->py() * photon.py()) - (incomingProton->pz() * photon.pz());
  double x = Q2 / (2. * dotProduct);

  // Summary: manually calculated Q2 seems to be a little high, while the x seems to fluctuate up and down.
  std::cout << "Manually calculated  Q2: " << Q2 << ", x: " << x << "\n";
  std::cout << "Extracted from HepMC Q2: " << m_q2 << ", x1: " << m_x1 << ", m_x2: " << m_x2 << "\n";
  */

  /*JetMap* jets = findNode::getClass<JetMap>(topNode, m_recoJetName);


  // interface to jets
  JetMap* jets = findNode::getClass<JetMap>(topNode, m_recoJetName);
  if (!jets)
  {
    cout
        << "WriteTree::process_event - Error can not find DST JetMap node "
        << m_recoJetName << endl;
    exit(-1);
  }

  // interface to tracks
  SvtxTrackMap* trackmap = findNode::getClass<SvtxTrackMap>(topNode, "SvtxTrackMap");
  if (!trackmap)
  {
    trackmap = findNode::getClass<SvtxTrackMap>(topNode, "TrackMap");
    if (!trackmap)
    {
      cout
          << "WriteTree::process_event - Error can not find DST trackmap node SvtxTrackMap" << endl;
      exit(-1);
    }
  }
  for (JetMap::Iter iter = jets->begin(); iter != jets->end(); ++iter)
  {
    Jet* jet = iter->second;
    assert(jet);

    bool eta_cut = (jet->get_eta() >= m_etaRange.first) and (jet->get_eta() <= m_etaRange.second);
    bool pt_cut = (jet->get_pt() >= m_ptRange.first) and (jet->get_pt() <= m_ptRange.second);
    if ((not eta_cut) or (not pt_cut))
    {
      if (Verbosity() >= WriteTree::VERBOSITY_MORE)
      {
        cout << "WriteTree::process_event() - jet failed acceptance cut: ";
        cout << "eta cut: " << eta_cut << ", ptcut: " << pt_cut << endl;
        cout << "jet eta: " << jet->get_eta() << ", jet pt: " << jet->get_pt() << endl;
        jet->identify();
      }
      continue;
    }

    // fill histograms
    assert(m_hInclusiveE);
    m_hInclusiveE->Fill(jet->get_e());
    assert(m_hInclusiveEta);
    m_hInclusiveEta->Fill(jet->get_eta());
    assert(m_hInclusivePhi);
    m_hInclusivePhi->Fill(jet->get_phi());

    // fill trees - jet spectrum
    Jet* truthjet = recoeval->max_truth_jet_by_energy(jet);

    m_id = jet->get_id();
    m_nComponent = jet->size_comp();
    m_eta = jet->get_eta();
    m_phi = jet->get_phi();
    m_e = jet->get_e();
    m_pt = jet->get_pt();

    m_truthID = -1;
    m_truthNComponent = -1;
    m_truthEta = NAN;
    m_truthPhi = NAN;
    m_truthE = NAN;
    m_truthPt = NAN;

    if (truthjet)
    {
      m_truthID = truthjet->get_id();
      m_truthNComponent = truthjet->size_comp();
      m_truthEta = truthjet->get_eta();
      m_truthPhi = truthjet->get_phi();
      m_truthE = truthjet->get_e();
      m_truthPt = truthjet->get_pt();
    }

    // fill trees - jet track matching
    m_nMatchedTrack = 0;

    for (SvtxTrackMap::Iter iter = trackmap->begin();
         iter != trackmap->end();
         ++iter)
    {
      SvtxTrack* track = iter->second;

      TVector3 v(track->get_px(), track->get_py(), track->get_pz());
      const double dEta = v.Eta() - m_eta;
      const double dPhi = v.Phi() - m_phi;
      const double dR = sqrt(dEta * dEta + dPhi * dPhi);

      if (dR < m_trackJetMatchingRadius)
      {
        //matched track to jet

        assert(m_nMatchedTrack < kMaxMatchedTrack);

        m_trackdR[m_nMatchedTrack] = dR;
        m_trackpT[m_nMatchedTrack] = v.Perp();

        ++m_nMatchedTrack;
      }

      if (m_nMatchedTrack >= kMaxMatchedTrack)
      {
        cout << "WriteTree::process_event() - reached max track that matching a jet. Quit iterating tracks" << endl;
        break;
      }

    }  //    for (SvtxTrackMap::Iter iter = trackmap->begin();

    m_T->Fill();
  }  //   for (JetMap::Iter iter = jets->begin(); iter != jets->end(); ++iter)
  */

  return Fun4AllReturnCodes::EVENT_OK;
}

