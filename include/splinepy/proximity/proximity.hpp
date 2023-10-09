#pragma once

#include <napf.hpp>

#include <splinepy/splines/helpers/properties.hpp>
#include <splinepy/splines/splinepy_base.hpp>
#include <splinepy/utils/arrays.hpp>
#include <splinepy/utils/grid_points.hpp>
#include <splinepy/utils/nthreads.hpp>
#include <splinepy/utils/print.hpp>

namespace splinepy::proximity {

/*!
 * A helper class to perform proximity operations for splines.
 *
 * Given a physical coordinate, this tries to find parametric coordinate that
 * maps to the closest physical coordinate. Often referred as
 * "point inversion". This is done by searching for a root of first derivative
 * of squared distance between mapped coordinate and query coordinate.
 * For detailed information, please take a look at splinepy python
 * documentation.
 */
class Proximity {
public:
  /// @brief array cloud that wraps data of Array
  using Cloud_ = napf::ArrayCloud<double, int>;
  /// @brief metric is L2 and this returns squared distance
  using Tree_ = napf::ArrayTree<double, double, int, 2>;

  using RealArray_ = splinepy::utils::Array<double>;
  using RealArray2D_ = splinepy::utils::Array<double, 2>;
  using RealArray3D_ = splinepy::utils::Array<double, 3>;
  using IndexArray_ = splinepy::utils::Array<int>;

protected:
  // helpee spline
  const splinepy::splines::SplinepyBase& spline_;

  // kdtree related variables
  splinepy::utils::CStyleArrayPointerGridPoints grid_points_;
  RealArray_ sampled_spline_;
  std::unique_ptr<Cloud_> cloud_;
  std::unique_ptr<Tree_> kdtree_;

public:
  /// Constructor. As a spline helper class, always need a spline.
  Proximity(const splinepy::splines::SplinepyBase& spline) : spline_(spline){};

  /*!
   * Plants a kdtree with given resolution.
   *
   * This needs to be built before BEFORE you request a proximity query with
   * `InitialGuess::kdTree`. This will always plant a new tree: at runtime, if
   * a finer tree is desired, you can plant it again.
   *
   * @param resolutions parameter space sampling resolution
   * @param n_thread number of threads to be used for sampling
   */
  void PlantNewKdTree(const int* resolutions, const int n_thread = 1) {
    const int para_dim = spline_.SplinepyParaDim();
    const int dim = spline_.SplinepyDim();

    // get parametric bounds
    RealArray_ parametric_bounds(para_dim * 2);
    spline_.SplinepyParametricBounds(parametric_bounds.data());

    // create grid_points
    grid_points_.SetUp(para_dim, parametric_bounds.data(), resolutions);

    const int n_queries = grid_points_.Size();

    // reallocate sampled spline
    sampled_spline_.Reallocate(n_queries * dim);

    // lambda function to allow n-thread execution
    auto sample_coordinates = [&](int begin, int end) {
      RealArray_ query(para_dim);
      double* query_data = query.data();

      for (int i{begin}; i < end; ++i) {
        grid_points_.IdToGridPoint(i, query_data);
        spline_.SplinepyEvaluate(query_data, &sampled_spline_[i * dim]);
      }
    };

    // n-thread execution for sampling
    splinepy::utils::NThreadExecution(sample_coordinates, n_queries, n_thread);

    // nanoflann supports concurrent build
    nanoflann::KDTreeSingleIndexAdaptorParams params{};
    params.n_thread_build = static_cast<
        decltype(nanoflann::KDTreeSingleIndexAdaptorParams::n_thread_build)>(
        (n_thread < 0) ? 0 : n_thread);

    // plant a new tree
    cloud_ = std::make_unique<Cloud_>(sampled_spline_.data(),
                                      sampled_spline_.size(),
                                      dim);
    kdtree_ = std::make_unique<Tree_>(dim, *cloud_, params);
  }

  /// @brief difference = spline(guess) - query. In current formulation, this is
  /// our objective function.
  /// @param guess
  /// @param query
  /// @param difference
  void GuessMinusQuery(const RealArray_& guess,
                       const RealArray_& query,
                       RealArray_& difference) const {
    // evaluate guess and sett to difference
    spline_.SplinepyEvaluate(guess.data(), difference.data());
    // subtract query from evaluated guess
    difference.Subtract(query);
  }

  void MakeInitialGuess(const RealArray_& goal, RealArray_& guess) const {
    if (!kdtree_) {
      // hate to be aggressive, but here it is.
      splinepy::utils::PrintAndThrowError("to use InitialGuess::Kdtree option,"
                                          "please first plant a kdtree.");
    }

    // good to go. ask the tree
    int id;
    double distance;
    kdtree_->knnSearch(goal.data(), 1 /* closest neighbor */, &id, &distance);

    grid_points_.IdToGridPoint(id, guess.data());
  }

  /*!
   * Builds RHS and fills spline_gradient, which is also required in LHS.
   * RHS is what's internally called as "df_dxi"
   *
   * @param[in] guess current parametric coordinate guess
   * @param[in] difference result of `GuessMinusQuery()`
   * @param[out] spline_gradient
   * @param[out] rhs
   */
  void FillSplineGradientAndRhs(const RealArray_& guess,
                                const RealArray_& difference,
                                RealArray2D_& spline_gradient,
                                RealArray_& rhs) const {
    const int para_dim = guess.size();
    const int dim = difference.size();

    IndexArray_ derivative_query(para_dim);
    derivative_query.Fill(0);

    // this is just to view and apply inner product
    RealArray_ gradient_row_view;
    gradient_row_view.SetShape(dim);

    for (int i{}; i < para_dim; ++i) {
      // set query - this should be all zero here
      derivative_query[i] = 1;

      // set row view
      gradient_row_view.SetData(&spline_gradient(i, 0));

      // derivative evaluation
      spline_.SplinepyDerivative(guess.data(),
                                 derivative_query.data(),
                                 gradient_row_view.data());

      // fill rhs_i and apply minus here already!
      rhs[i] = -2. * difference.InnerProduct(gradient_row_view);

      // reset query to zero
      derivative_query[i] = 0;
    }
  }

  /*!
   * Builds LHS
   *
   * @param[in] guess
   * @param[in] difference
   * @param[in] spline_gradient
   * @param[out] lhs
   */
  void FillLhs(const RealArray_& guess,
               const RealArray_& difference,
               const RealArray2D_& spline_gradient_AAt,
               RealArray3D_& spline_hessian,
               RealArray2D_& lhs) const {
    const int para_dim = guess.size();
    const int dim = difference.size();

    IndexArray_ derivative_query(para_dim);
    derivative_query.Fill(0);

    // derivative result is a view to the hessian array
    RealArray_ derivative;
    derivative.SetShape(dim);

    // lambda to compute each element
    auto compute = [&](const int& i, const int& j) {
      // adjust derivative query
      ++derivative_query[i];
      ++derivative_query[j];

      // adjust view on hessian array
      derivative.SetData(&spline_hessian(i, j, 0));

      // compute
      spline_.SplinepyDerivative(guess.data(),
                                 derivative_query.data(),
                                 derivative.data());

      // fill lhs_ij
      lhs(i, j) =
          2.
          * (difference.InnerProduct(derivative) + spline_gradient_AAt(i, j));

      // reset derivative query
      --derivative_query[i];
      --derivative_query[j];
    };

    for (int i{}; i < para_dim; ++i) {
      for (int j{i + 1}; j < para_dim; ++j) {
        // upper triangle without diagonal
        compute(i, j);
        // copy to the lower
        lhs(j, i) = lhs(i, j);
        // copy spline hessian - maybe not?
        std::copy_n(derivative.data(), dim, &spline_hessian(j, i, 0));
      }
      // diagonal
      compute(i, i);
    }
  }

  /// @brief First order fall back
  void FirstOrderFallBack() {}

  /// @brief Given physical coordinate, finds closest parametric coordinate.
  /// Always takes initial guess based on kdtree.
  ///
  /// @param[in] query
  /// @param[in] tolerance
  /// @param[in] max_iterations
  /// @param[in] aggressive_bounds
  /// @param[out] final_guess (para_dim)
  /// @param[out] nearest (dim)
  /// @param[out] nearest_minus_query (dim)
  /// @param[out] distance
  /// @param[out] convergence_norm
  /// @param[out] first_derivatives (para_dim x dim)
  /// @param[out] second_derivatives (para_dim x para_dim x dim)
  void VerboseQuery(const double* query,
                    const double& tolerance,
                    const int& max_iterations,
                    const bool aggressive_bounds,
                    double* final_guess,
                    double* nearest /* spline(final_guess) */,
                    double* nearest_minus_query /* difference */,
                    double& distance,
                    double& convergence_norm,
                    double* first_derivatives /* spline jacobian */,
                    double* second_derivatives /* spline hessian */) const {

    const int para_dim = spline_.SplinepyParaDim();
    const int dim = spline_.SplinepyDim();

    // allocate aux real arrays
    RealArray2D_ lhs(para_dim, para_dim);
    RealArray_ rhs(para_dim);
    RealArray_ delta_guess(para_dim);
    RealArray_ difference(dim);
    RealArray2D_ spline_gradient(para_dim, dim);
    RealArray2D_ spline_gradient_AAt(para_dim, para_dim);
    RealArray_ current_guess(para_dim);
    RealArray_ current_phys(dim);
    RealArray2D_ search_bounds(2, para_dim);

    // view arrays
    RealArray_ query_view(query, dim);
    RealArray_ spline_gradient_view(spline_gradient, para_dim, dim);
    // get pointers to beginning of each bound
    double* lower_bound = search_bounds.begin();
    double* upper_bound = lower_bound + dim;

    // allocate index arrays
    IndexArray_ clipped(para_dim);
    IndexArray_ previous_clipped(para_dim);

    // values
    double current_distance{};
    double previous_norm{}, current_norm{};

    // search_bounds is parametric bounds here
    spline_.SplinepyParametricBounds(search_bounds.data());

    // initial guess
    // for verbose, we don't return right away even this is the best guess
    // already, so that we can fill out all the other infos.
    MakeInitialGuess(query_view, current_guess);

    // Let's try aggressive search bounds
    if (aggressive_bounds) {
      // you need to be sure that you have sampled your spline fine enough
      for (int i{}; i < para_dim; ++i) {
        // adjust lower (0) and upper (1) bounds aggressively
        // but of course, not so aggressive that it is out of bound.
        search_bounds(0, i) =
            std::max(search_bounds(0, i),
                     current_guess[i] - grid_points_.step_size_[i]);
        search_bounds(1, i) =
            std::min(search_bounds(1, i),
                     current_guess[i] + grid_points_.step_size_[i]);
      }
    }
    const int max_iter = max_iterations < 0 ? para_dim * 20 : max_iterations;

    // build systems to solve
    GuessMinusQuery(current_guess, query_view, difference);
    FillSplineGradientAndRhs(current_guess, difference, spline_gradient, rhs);

    // 0 iteration returns initial guess.
    // compute rest of verbose info here.
    if (max_iterations == 0) {
      current_phys = spline_(current_guess);
      splinepy::utils::FirstMinusSecondEqualsThird(current_phys,
                                                   query,
                                                   difference);
      current_distance = splinepy::utils::NormL2(difference);
      current_norm = std::abs(splinepy::utils::NormL2(rhs));
    }

    // newton iterations
    for (int i{}; i < max_iter; ++i) {
      // lhs
      FillLhs(current_guess, difference, spline_gradient, lhs);

      // GaussWithPivot may swap and modify enties of all the input
      // -> can't use lhs and rhs afterwards, and we don't need them.
      // -> solver_skip_mask and delta_guess is reordered to rewind swaps
      splinepy::utils::GaussWithPivot(lhs, rhs, solver_skip_mask, delta_guess);
      // Update
      splinepy::utils::AddSecondToFirst(current_guess, delta_guess);
      // Clip
      splinepy::utils::Clip(search_bounds, current_guess, clipped);
      // check distance
      current_phys = spline_(current_guess);
      splinepy::utils::FirstMinusSecondEqualsThird(current_phys,
                                                   query,
                                                   difference);
      current_distance = splinepy::utils::NormL2(difference);
      // assemble rhs, check norm
      FillSplineGradientAndRhs(current_guess, difference, spline_gradient, rhs);
      current_norm = splinepy::utils::NormL2(rhs);

      // convergence check
      if (std::abs(previous_norm - current_norm) < tolerance
          || current_distance < tolerance) {
        break;
      }
      // set solver skip mask if clipping happened twice at the same place.
      if (previous_clipped == clipped) {
        solver_skip_mask = clipped;
        // if skip mask is on for all entries, return now.
        if (splinepy::utils::NonZeros(solver_skip_mask)
            == SplineType::kParaDim) {
          // current_guess should be clipped at this point.
          break;
        }
      }

      // we are here because it didn't converge. prepare next round
      previous_norm = current_norm;
      std::swap(previous_clipped, clipped);
    }
    // write return values - 7 args
    distance = current_distance;     /* 1 */
    convergence_norm = current_norm; /* 2 */
    typename SplineType::Derivative_ derivative_query;
    double der; /* to accommodate different kind of derivative types */
    using DerivativeValueType = typename SplineType::Derivative_::value_type;
    for (int i{}; i < SplineType::kParaDim; ++i) {
      final_guess[i] = static_cast<double>(current_guess[i]); /* 3 */
      for (int j{i}; j < SplineType::kParaDim; ++j) {
        derivative_query.fill(DerivativeValueType{0});
        ++derivative_query[i];
        ++derivative_query[j];
        const auto derivative = spline_(current_guess, derivative_query);
        for (int k{}; k < SplineType::kDim; ++k) {
          if constexpr (std::is_scalar<decltype(derivative)>::value) {
            der = derivative;
          } else {
            der = static_cast<double>(derivative[k]);
          }
          // spline hessian
          second_derivatives[(i * SplineType::kParaDim * SplineType::kDim)
                             + (j * SplineType::kDim) + k] = der; /* 4 */
          // symmetric part
          if (i != j) {
            second_derivatives[(j * SplineType::kParaDim * SplineType::kDim)
                               + (i * SplineType::kDim) + k] = der;
          }

          // ones that don't need extra para_dim loop
          if (i == 0 /* j starts with 0 */) {
            first_derivatives[j * SplineType::kDim + k] =
                spline_gradient[j][k]; /* 5 */
            // ones that don't need extra extra para_dim loop
            // => pure dim loop
            if (j == 0) {
              double cur_phys;
              if constexpr (std::is_scalar_v<decltype(current_phys)>) {
                nearest[k] = current_phys;
              } else {
                nearest[k] = static_cast<double>(current_phys[k]); /* 6 */
              }
              nearest_minus_query[k] = difference[k]; /* 7 */
            }
          }
        }
      }
    }
  }
};

} // namespace splinepy::proximity
