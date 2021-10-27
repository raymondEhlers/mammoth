
#include <vector>
#include <numeric>

#include <TSystem.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TFile.h>
#include <TCanvas.h>

#include <LHAPDF/LHAPDF.h>

void testLHAPDF()
{
    gSystem->Load("libLHAPDF");
    //this->pdf = std::make_unique<LHAPDF::PDF>(LHAPDF::mkPDF("EPPS16nlo_CT14nlo_Au197"));
    // We get the raw pointer for LHAPDF, so we want to encapsulate it in the unique_ptr
    std::unique_ptr<LHAPDF::PDF> pdf, npdf;
    pdf.reset(LHAPDF::mkPDF("CT14nlo"));
    npdf.reset(LHAPDF::mkPDF("EPPS16nlo_CT14nlo_Au197"));

    int nLogBins = 400;
    std::vector<double> logBins(nLogBins + 1);
    double xlogmin = std::log10(1e-4);
    double xlogmax = std::log10(1);
    double dlogx   = (xlogmax-xlogmin)/(static_cast<double>(nLogBins));
    for (int i=0; i<=nLogBins; i++) {
        double xlog = xlogmin + i*dlogx;
        logBins.at(i) = std::exp(std::log(10) * xlog);
        //std::cout << logBins[i] << " ";
    }

    std::vector<double> q2(100);
    std::iota(std::begin(q2), std::end(q2), 1);

    TH2D ratio("ratio", "ratio;x;q2", nLogBins, logBins.data(), 100, 0.5, 100.5);
    TH2D nPDFAlone("nPDFAlone", "nPDFAlone;x;q2", nLogBins, logBins.data(), 100, 0.5, 100.5);

    // 2 == up quark
    int struckQuarkPDGCode = 2;

    for (auto q2Val : q2) {
        for (auto xVal : logBins) {
            double weightNPDF = npdf->xfxQ2(struckQuarkPDGCode, xVal, q2Val);
            double weightPDF = pdf->xfxQ2(struckQuarkPDGCode, xVal, q2Val);
            ratio.Fill(xVal, q2Val, weightNPDF / weightPDF);
            nPDFAlone.Fill(xVal, q2Val, weightNPDF);
        }
    }

    TCanvas c("c");
    ratio.Draw("colz");
    c.SetLogx();
    c.SetLogz();
    c.SaveAs("pdfTest.pdf");
    c.Clear();

    auto h = ratio.ProjectionX("proj", 10, 10);
    h->GetXaxis()->SetRangeUser(1e-4, 0.9);
    h->GetYaxis()->SetTitle("eA/ep");
    h->GetYaxis()->SetRangeUser(0, 1.5);
    h->Draw("HIST");
    c.SetLogx();
    c.SaveAs("pdfTestProj.pdf");
    c.Clear();

    // Alone
    nPDFAlone.Draw("colz");
    c.SetLogx();
    c.SetLogz();
    c.SaveAs("pdfTestNPDFAlone.pdf");
    c.Clear();

    auto h2 = nPDFAlone.ProjectionX("proj2", 10, 10);
    h2->GetXaxis()->SetRangeUser(1e-4, 0.9);
    h2->GetYaxis()->SetTitle("xf(x)");
    h2->GetYaxis()->SetRangeUser(0, 1.5);
    h2->Draw("HIST");
    c.SetLogx();
    c.SaveAs("pdfTestNPDFAloneProj.pdf");
    c.Clear();

    TFile fOut("pdfTest.root", "RECREATE");
    fOut.cd();
    ratio.Write();
    nPDFAlone.Write();
    fOut.Close();
}
