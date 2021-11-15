#pragma once

#include <map>
#include <string>
#include <vector>

namespace alice {
namespace fastsim {
    /**
      * @brief Fast sim based on tracking efficiency parametrization
      *
      * This is, at best, right as a 0th order correction, but it's good enough to play around with.
      * These parametrizations are ported over from `AliAnalysisTaskEmcalJetHUtils` in AliPhysics.
      */

    /**
     * @brief Efficiency errors
     *
     */
    enum class Error_t {
        kInvalidCentrality,
        kPeriodNotImplemented,
        kInvalidPeriod,
        kTrackInputSizeMismatch,
    };

    /**
     * @enum Period_t
     * @brief Identify the beam type and period that is being analyzed.
     */
    enum class Period_t {
        kDisabled = -1,     //!<! No identifier
        kLHC11h = 0,        //!<! Run1 PbPb - LHC11h
        kLHC15o = 1,        //!<! Run2 PbPb - LHC15o
        kLHC18qr = 2,       //!<! Run2 PbPb - LHC18{q,r}
        kLHC11a = 3,        //!<! Run1 pp - LHC11a (2.76 TeV)
        kpA = 4,            //!<! Generic pA
        kpp = 5,            //!<! Generic pp
    };

    /**
     * @brief Map from name to efficiency period
     *
     * For use with the YAML config and elsewhere where strings are more convenient.
     *
     */
    const std::map<std::string, Period_t> periodNameMap = {
        { "disabled", Period_t::kDisabled},
        { "LHC11h", Period_t::kLHC11h},
        { "LHC15o", Period_t::kLHC15o},
        { "LHC18q", Period_t::kLHC18qr},
        { "LHC18r", Period_t::kLHC18qr},
        { "LHC11a", Period_t::kLHC11a},
        { "pA", Period_t::kpA },
        { "pp", Period_t::kpp }
    };

    /**
     * @brief Provide the tracking efficiency as a function of track pt, track eta, and centrality.
     *
     * Here, we use this function as a 0th order fast simulation, since we expect the tracking efficiency
     * to be the dominant effect of the efficiency.
     *
     * NOTE: This function relies on the user taking advantage of the centrality binning in ``AliAnalysisTaskEmcal``.
     *       If it's not used, the proper centrality dependent efficiency may not be applied.
     *
     * @param[in] trackPt Track pt
     * @param[in] trackEta Track eta
     * @param[in] centralityBin Centrality bin of the current event.
     * @param[in] period Identifies the data taking period to select the efficiency parametrization.
     * @param[in] taskName Name of the task which is calling this function (for logging purposes).
     * @returns The efficiency of measuring the given single track.
     */
    double TrackingEfficiencyByPeriod(
        const double trackPt, const double trackEta,
        const int centralityBin,
        const Period_t period,
        const std::string &taskName);
}
}