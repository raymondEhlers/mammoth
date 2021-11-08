import ROOT

f = ROOT.TFile.Open("output_JetObservables.root")
h = f.Get("nBackwardHadrons_full_true")
print(f"h: {h}")
h.GetXaxis().SetTitle("n hadrons")

c = ROOT.TCanvas("c", "c")

#h.Scale(1/10.0)
#h.Scale(1/1000.0)
# This gives 0 for some reason. Because fuck root
#h.Scale(1./h.Integral(1, -1))
h.Draw()
print(f"Mean: {h.GetMean()}")
print(f"Integral: {h.Integral(1, -1)}")

#c.SetLogy()

c.SaveAs("backwardsHadrons.pdf")
