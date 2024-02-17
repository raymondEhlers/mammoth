#pragma once

#include <vector>
#include <string>

#include <fun4all/SubsysReco.h>

#include <TTree.h>

class WriteTree : public SubsysReco
{
 public:
  WriteTree(const std::string &outputfilename = "writeTree.root");

  virtual ~WriteTree();

  int Init(PHCompositeNode *topNode);
  int InitRun(PHCompositeNode *topNode);
  int process_event(PHCompositeNode *topNode);
  int End(PHCompositeNode *topNode);

 private:
  std::string m_outputFileName;

  //! Output Tree variables
  TTree *m_T;

  int m_event;
  float m_x1;
  float m_x2;
  float m_q2;
  std::vector<int> m_particleID;
  std::vector<int> m_particleStatus;
  std::vector<float> m_pt;
  std::vector<float> m_eta;
  std::vector<float> m_phi;
  std::vector<float> m_E;
};
