#pragma once

/**
 * Tools and utilities for analysis
 *
 * NOTE:
 *    The ROOT utils are taken from the master in ROOT as of Nov 2023,
 *    so corresponding to approximately 6.28.06
 *
 * @author: Raymond Ehlers <raymond.ehlers@cern.ch>, LBL/UCB
 */

#include <string>
#include <vector>

namespace mammoth {

namespace analysis {

/**
 * @brief Smooth an array using a running median
 *
 * This is a port of the ROOT implementation of TH1::SmoothArray, which itself
 * is a translation of Hbook routine `hsmoof.F`. It's all based on algorithm
 * 353QH by J. Friedman, and is described in [Proc. of the 1974 CERN School of
 * Computing, Norway, 11-24 August, 1974](https://cds.cern.ch/record/186223).
 * See also Section 4.2 in [J. Friedman, Data Analysis Techniques for High
 * Energy
 * Physics](https://www.slac.stanford.edu/pubs/slacreports/reports16/slac-r-176.pdf).
 *
 * @param[in] array Array to smooth.
 * @param[in] nTimes Number of times to smooth the array. Defaults to 1.
 */
template <typename T>
std::vector<T> smoothArray(const std::vector<T> &array, int nTimes = 1);

namespace detail {
using Long64_t = long long;

/*
 * From TMathBase.h
 */
template <typename T> struct CompareDesc {

  CompareDesc(T d) : fData(d) {}

  template <typename Index> bool operator()(Index i1, Index i2) {
    return *(fData + i1) > *(fData + i2);
  }

  T fData;
};

/*
 * From TMathBase.h
 */
template <typename T> struct CompareAsc {

  CompareAsc(T d) : fData(d) {}

  template <typename Index> bool operator()(Index i1, Index i2) {
    return *(fData + i1) < *(fData + i2);
  }

  T fData;
};

/*
 * Port of ROOT::TMath::Sort
 *
 * ROOT documentation is below:
 *
 * Sort the n elements of the  array a of generic templated type Element.
 * In output the array index of type Index contains the indices of the sorted
 * array. If down is false sort in increasing order (default is decreasing
 * order).
 *
 * NOTE that the array index must be created with a length >= n
 * before calling this function.
 * NOTE also that the size type for n must be the same type used for the index
 * array (templated type Index)
 */
template <typename Element, typename Index>
void root_tmath_sort(Index n, const Element *a, Index *index, bool down = true);

/*
 * Port of TMath::KOrdStat.
 *
 * Needed for TMath::Median.  ROOT documentation is below:
 *
 * Returns k_th order statistic of the array a of size n
 * (k_th smallest element out of n elements).
 *
 * C-convention is used for array indexing, so if you want
 * the second smallest element, call KOrdStat(n, a, 1).
 *
 * If work is supplied, it is used to store the sorting index and
 * assumed to be >= n. If work=0, local storage is used, either on
 * the stack if n < kWorkMax or on the heap for n >= kWorkMax.
 * Note that the work index array will not contain the sorted indices but
 * all indices of the smaller element in arbitrary order in work[0,...,k-1] and
 * all indices of the larger element in arbitrary order in work[k+1,..,n-1]
 * work[k] will contain instead the index of the returned element.
 *
 * Taken from "Numerical Recipes in C++" without the index array
 * implemented by Anna Khreshuk.
 *
 * See also the declarations at the top of this file
 */
template <class Element, typename Size>
Element KOrdStat(Size n, const Element *a, Size k, Size *work = nullptr);

/*
 * Port of ROOT::TMath::Median.
 *
 * ROOT documentation is below:
 *
 * Returns the median of the array a where each entry i has weight w[i] .
 * Both arrays have a length of at least n . The median is a number obtained
 * from the sorted array a through
 *
 * median = (a[jl]+a[jh])/2.  where (using also the sorted index on the array w)
 *
 * sum_i=0,jl w[i] <= sumTot/2
 * sum_i=0,jh w[i] >= sumTot/2
 * sumTot = sum_i=0,n w[i]
 *
 * If w=0, the algorithm defaults to the median definition where it is
 * a number that divides the sorted sequence into 2 halves.
 * When n is odd or n > 1000, the median is kth element k = (n + 1) / 2.
 * when n is even and n < 1000the median is a mean of the elements k = n/2 and k
 * = n/2 + 1.
 *
 * If the weights are supplied (w not 0) all weights must be >= 0
 *
 * If work is supplied, it is used to store the sorting index and assumed to be
 * >= n . If work=0, local storage is used, either on the stack if n < kWorkMax
 * or on the heap for n >= kWorkMax .
 */
template <typename T>
double root_tmath_median(Long64_t n, const T *a, const double *w = nullptr,
                         Long64_t *work = nullptr);

} // namespace detail

} // namespace analysis

} // namespace mammoth

template <typename Element, typename Index>
void mammoth::analysis::detail::root_tmath_sort(Index n, const Element *a,
                                                Index *index, bool down) {
  for (Index i = 0; i < n; i++) {
    index[i] = i;
  }
  if (down)
    std::sort(index, index + n, CompareDesc<const Element *>(a));
  else
    std::sort(index, index + n, CompareAsc<const Element *>(a));
}

template <class Element, typename Size>
Element mammoth::analysis::detail::KOrdStat(Size n, const Element *a, Size k,
                                            Size *work) {

  const int kWorkMax = 100;

  typedef Size Index;

  bool isAllocated = false;
  Size i, ir, j, l, mid;
  Index arr;
  Index *ind;
  Index workLocal[kWorkMax];
  Index temp;

  if (work) {
    ind = work;
  } else {
    ind = workLocal;
    if (n > kWorkMax) {
      isAllocated = true;
      ind = new Index[n];
    }
  }

  for (Size ii = 0; ii < n; ii++) {
    ind[ii] = ii;
  }
  Size rk = k;
  l = 0;
  ir = n - 1;
  for (;;) {
    if (ir <= l + 1) { // active partition contains 1 or 2 elements
      if (ir == l + 1 && a[ind[ir]] < a[ind[l]]) {
        temp = ind[l];
        ind[l] = ind[ir];
        ind[ir] = temp;
      }
      Element tmp = a[ind[rk]];
      if (isAllocated)
        delete[] ind;
      return tmp;
    } else {
      mid = (l + ir) >> 1; // choose median of left, center and right
      {
        temp = ind[mid];
        ind[mid] = ind[l + 1];
        ind[l + 1] = temp;
      }                           // elements as partitioning element arr.
      if (a[ind[l]] > a[ind[ir]]) // also rearrange so that a[l]<=a[l+1]
      {
        temp = ind[l];
        ind[l] = ind[ir];
        ind[ir] = temp;
      }

      if (a[ind[l + 1]] > a[ind[ir]]) {
        temp = ind[l + 1];
        ind[l + 1] = ind[ir];
        ind[ir] = temp;
      }

      if (a[ind[l]] > a[ind[l + 1]]) {
        temp = ind[l];
        ind[l] = ind[l + 1];
        ind[l + 1] = temp;
      }

      i = l + 1; // initialize pointers for partitioning
      j = ir;
      arr = ind[l + 1];
      for (;;) {
        do
          i++;
        while (a[ind[i]] < a[arr]);
        do
          j--;
        while (a[ind[j]] > a[arr]);
        if (j < i)
          break; // pointers crossed, partitioning complete
        {
          temp = ind[i];
          ind[i] = ind[j];
          ind[j] = temp;
        }
      }
      ind[l + 1] = ind[j];
      ind[j] = arr;
      if (j >= rk)
        ir = j - 1; // keep active the partition that
      if (j <= rk)
        l = i; // contains the k_th element
    }
  }
}

template <typename T>
double mammoth::analysis::detail::root_tmath_median(Long64_t n, const T *a,
                                                    const double *w,
                                                    Long64_t *work) {

  const int kWorkMax = 100;

  if (n <= 0 || !a)
    return 0;
  bool isAllocated = false;
  double median;
  Long64_t *ind;
  Long64_t workLocal[kWorkMax];

  if (work) {
    ind = work;
  } else {
    ind = workLocal;
    if (n > kWorkMax) {
      isAllocated = true;
      ind = new Long64_t[n];
    }
  }

  if (w) {
    double sumTot2 = 0;
    for (int j = 0; j < n; j++) {
      if (w[j] < 0) {
        throw std::runtime_error(
            std::string("root_tmath_Median: w[" + std::to_string(j) +
                        "] = " + std::to_string(w[j]) + " < 0 ?!"));
        if (isAllocated)
          delete[] ind;
        return 0;
      }
      sumTot2 += w[j];
    }

    sumTot2 /= 2.;

    detail::root_tmath_sort(n, a, ind, false);

    double sum = 0.;
    int jl;
    for (jl = 0; jl < n; jl++) {
      sum += w[ind[jl]];
      if (sum >= sumTot2)
        break;
    }

    int jh;
    sum = 2. * sumTot2;
    for (jh = n - 1; jh >= 0; jh--) {
      sum -= w[ind[jh]];
      if (sum <= sumTot2)
        break;
    }

    median = 0.5 * (a[ind[jl]] + a[ind[jh]]);

  } else {

    if (n % 2 == 1)
      median = KOrdStat(n, a, n / 2, ind);
    else {
      median =
          0.5 * (KOrdStat(n, a, n / 2 - 1, ind) + KOrdStat(n, a, n / 2, ind));
    }
  }

  if (isAllocated)
    delete[] ind;
  return median;
}

template <typename T>
std::vector<T> mammoth::analysis::smoothArray(const std::vector<T> &array,
                                              int nTimes) {
  if (array.size() < 3) {
    throw std::out_of_range("Need at least 3 points for smoothing: n = " +
                            std::to_string(array.size()));
  }
  // RJE: NOTE: Originally, `xx` is a double * containing the array that is passed into
  //            this function. Since I didn't want to deal with a raw pointer, I changed
  //            how the data is passed, although I changed the algorithm as little as possible
  //            beyond that. To make this workable, we'll copy the input array into `xx`,
  //            which is now a `std::vector<double>`. It's changed in place, so we need to
  //            return it explicitly.
  int nn = array.size();
  std::vector<T> xx(nn);
  // RJE: NOTE: For this implement, It's important to copy the array values into
  //            xx before running the algorithm. Otherwise, we get the wrong answer.
  //            In this case, we're copying the array twice, but this code isn't
  //            performance critical, so it's not a big deal. (and definitely not worth
  //            changing the underlying ROOT algorithm to avoid copying!)
  std::copy(array.begin(), array.end(), xx.begin());

  int ii;
  std::array<double, 3> hh{};

  std::vector<double> yy(nn);
  std::vector<double> zz(nn);
  std::vector<double> rr(nn);

  for (int pass = 0; pass < nTimes; pass++) {
    // first copy original data into temp array
    // RJE: Minor changed from copying xx via offsets to using iterators. It's safer.
    std::copy(xx.begin(), xx.end(), zz.begin());

    for (int noent = 0; noent < 2; ++noent) { // run algorithm two times

      //  do 353 i.e. running median 3, 5, and 3 in a single loop
      for (int kk = 0; kk < 3; kk++) {
        std::copy(zz.begin(), zz.end(), yy.begin());
        int medianType = (kk != 1) ? 3 : 5;
        int ifirst = (kk != 1) ? 1 : 2;
        int ilast = (kk != 1) ? nn - 1 : nn - 2;
        // nn2 = nn - ik - 1;
        //  do all elements beside the first and last point for median 3
        //   and first two and last 2 for median 5
        for (ii = ifirst; ii < ilast; ii++) {
          zz[ii] =
              detail::root_tmath_median(medianType, yy.data() + ii - ifirst);
        }

        if (kk == 0) { // first median 3
          // first point
          hh[0] = zz[1];
          hh[1] = zz[0];
          hh[2] = 3 * zz[1] - 2 * zz[2];
          zz[0] = detail::root_tmath_median(3, hh.data());
          // last point
          hh[0] = zz[nn - 2];
          hh[1] = zz[nn - 1];
          hh[2] = 3 * zz[nn - 2] - 2 * zz[nn - 3];
          zz[nn - 1] = detail::root_tmath_median(3, hh.data());
        }

        if (kk == 1) { //  median 5
          // second point with window length 3
          zz[1] = detail::root_tmath_median(3, yy.data());
          // second-to-last point with window length 3
          zz[nn - 2] = detail::root_tmath_median(3, yy.data() + nn - 3);
        }

        // In the third iteration (kk == 2), the first and last point stay
        // the same (see paper linked in the documentation).
      }

      std::copy(zz.begin(), zz.end(), yy.begin());

      // quadratic interpolation for flat segments
      for (ii = 2; ii < (nn - 2); ii++) {
        if (zz[ii - 1] != zz[ii])
          continue;
        if (zz[ii] != zz[ii + 1])
          continue;
        const double tmp0 = zz[ii - 2] - zz[ii];
        const double tmp1 = zz[ii + 2] - zz[ii];
        if (tmp0 * tmp1 <= 0)
          continue;
        int jk = 1;
        if (std::abs(tmp0) > std::abs(tmp0))
          jk = -1;
        yy[ii] = -0.5 * zz[ii - 2 * jk] + zz[ii] / 0.75 + zz[ii + 2 * jk] / 6.;
        yy[ii + jk] = 0.5 * (zz[ii + 2 * jk] - zz[ii - 2 * jk]) + zz[ii];
      }

      // running means
      // std::copy(zz.begin(), zz.end(), yy.begin());
      for (ii = 1; ii < nn - 1; ii++) {
        zz[ii] = 0.25 * yy[ii - 1] + 0.5 * yy[ii] + 0.25 * yy[ii + 1];
      }
      zz[0] = yy[0];
      zz[nn - 1] = yy[nn - 1];

      if (noent == 0) {

        // save computed values
        std::copy(zz.begin(), zz.end(), rr.begin());

        // COMPUTE  residuals
        for (ii = 0; ii < nn; ii++) {
          zz[ii] = xx[ii] - zz[ii];
        }
      }

    } // end loop on noent

    // RJE: Switch to std
    auto xmin = std::min_element(xx.begin(), xx.end());
    // double xmin = TMath::MinElement(nn,xx);
    for (ii = 0; ii < nn; ii++) {
      if (*xmin < 0)
        xx[ii] = rr[ii] + zz[ii];
      // make smoothing defined positive - not better using 0 ?
      else
        xx[ii] = std::max((rr[ii] + zz[ii]), 0.0);
    }
  }
  return xx;
}
