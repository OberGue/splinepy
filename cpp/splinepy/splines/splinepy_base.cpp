#include <memory>
#include <vector>

#include <splinepy/splines/create/create_bezier.hpp>
#include <splinepy/splines/create/create_bspline.hpp>
#include <splinepy/splines/create/create_nurbs.hpp>
#include <splinepy/splines/create/create_rational_bezier.hpp>
#include <splinepy/splines/splinepy_base.hpp>
#include <splinepy/utils/print.hpp>

namespace splinepy::splines {

std::shared_ptr<SplinepyBase> SplinepyBase::SplinepyCreate(
    const int para_dim,
    const int dim,
    const int* degrees,
    const std::vector<std::vector<double>>* knot_vectors,
    const double* control_points,
    const double* weights) {
  if (!degrees || !control_points) {
    splinepy::utils::PrintAndThrowError(
        "Not Enough information to create any spline.");
  }

  if (!knot_vectors) {
    // at least we need degrees and cps.
    // @jzwar this would be a good place to check valid input

    if (!weights) {
      return SplinepyCreateBezier(para_dim, dim, degrees, control_points);
    } else {
      return SplinepyCreateRationalBezier(para_dim,
                                          dim,
                                          degrees,
                                          control_points,
                                          weights);
    }
  } else {
    if (!weights) {
      return SplinepyCreateBSpline(para_dim,
                                   dim,
                                   degrees,
                                   knot_vectors,
                                   control_points);
    } else {
      return SplinepyCreateNurbs(para_dim,
                                 dim,
                                 degrees,
                                 knot_vectors,
                                 control_points,
                                 weights);
    }
  }
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyCreateBezier(const int para_dim,
                                   const int dim,
                                   const int* degrees,
                                   const double* control_points) {
  return splinepy::splines::create::CreateBezier(para_dim,
                                                 dim,
                                                 degrees,
                                                 control_points);
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyCreateRationalBezier(const int para_dim,
                                           const int dim,
                                           const int* degrees,
                                           const double* control_points,
                                           const double* weights) {
  return splinepy::splines::create::CreateRationalBezier(para_dim,
                                                         dim,
                                                         degrees,
                                                         control_points,
                                                         weights);
}

std::shared_ptr<SplinepyBase> SplinepyBase::SplinepyCreateBSpline(
    const int para_dim,
    const int dim,
    const int* degrees,
    const std::vector<std::vector<double>>* knot_vectors,
    const double* control_points) {
  return splinepy::splines::create::CreateBSpline(para_dim,
                                                  dim,
                                                  degrees,
                                                  knot_vectors,
                                                  control_points);
}

std::shared_ptr<SplinepyBase> SplinepyBase::SplinepyCreateNurbs(
    const int para_dim,
    const int dim,
    const int* degrees,
    const std::vector<std::vector<double>>* knot_vectors,
    const double* control_points,
    const double* weights) {
  return splinepy::splines::create::CreateNurbs(para_dim,
                                                dim,
                                                degrees,
                                                knot_vectors,
                                                control_points,
                                                weights);
}

bool SplinepyBase::SplinepySplineNameMatches(const SplinepyBase& a,
                                             const SplinepyBase& b,
                                             const std::string description,
                                             const bool raise) {
  if (a.SplinepySplineName() != b.SplinepySplineName()) {
    if (raise) {
      splinepy::utils::PrintAndThrowError(description,
                                          "Spline name mismatch -"
                                          "Spline0:",
                                          a.SplinepySplineName(),
                                          "/",
                                          "Spline1:",
                                          b.SplinepySplineName());
    }
    return false;
  }
  return true;
}

bool SplinepyBase::SplinepyParaDimMatches(const SplinepyBase& a,
                                          const SplinepyBase& b,
                                          const std::string description,
                                          const bool raise) {
  if (a.SplinepyParaDim() != b.SplinepyParaDim()) {
    if (raise) {
      splinepy::utils::PrintAndThrowError(
          description,
          "Spline parametric dimension mismatch - "
          "Spline0:",
          a.SplinepyParaDim(),
          "/",
          "Spline1:",
          b.SplinepyParaDim());
    }
    return false;
  }
  return true;
}

bool SplinepyBase::SplinepyDimMatches(const SplinepyBase& a,
                                      const SplinepyBase& b,
                                      const std::string description,
                                      const bool raise) {
  if (a.SplinepyDim() != b.SplinepyDim()) {
    if (raise) {
      splinepy::utils::PrintAndThrowError(
          description,
          "Spline parametric dimension mismatch - "
          "Spline0:",
          a.SplinepyDim(),
          "/",
          "Spline1:",
          b.SplinepyDim());
    }
    return false;
  }
  return true;
}

void SplinepyBase::SplinepyParametricBounds(double* para_bounds) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyParametricBounds not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyControlMeshResolutions(int* control_mesh_res) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyControlMeshResolutions not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyGrevilleAbscissae(double* greville_abscissae,
                                             const int& i_para_dim) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyGrevilleAbscissae not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyEvaluate(const double* para_coord,
                                    double* evaluated) const {
  splinepy::utils::PrintAndThrowError("SplinepyEvaluate not implemented for",
                                      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyDerivative(const double* para_coord,
                                      const int* orders,
                                      double* derived) const {
  splinepy::utils::PrintAndThrowError("SplinepyDerivative not implemented for",
                                      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyJacobian(const double* para_coord,
                                    double* jacobians) const {
  splinepy::utils::PrintAndThrowError("SplinepyJacobian not implemented for",
                                      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyBasis(const double* para_coord,
                                 double* basis) const {
  splinepy::utils::PrintAndThrowError("SplinepyBasis not implemented for",
                                      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyBasisDerivative(const double* para_coord,
                                           const int* order,
                                           double* basis) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyBasisDerivative not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepySupport(const double* para_coord,
                                   int* support) const {
  splinepy::utils::PrintAndThrowError("SplinepySupport not implemented for",
                                      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyBasisAndSupport(const double* para_coord,
                                           double* basis,
                                           int* support) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyBasisAndSupport not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyBasisDerivativeAndSupport(const double* para_coord,
                                                     const int* orders,
                                                     double* basis,
                                                     int* support) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyBasisDerivativeAndSupport not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyPlantNewKdTreeForProximity(const int* resolutions,
                                                      const int& nthreads) {
  splinepy::utils::PrintAndThrowError(
      "SplinepyPlantNewKdTreeForProximity not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyVerboseProximity(const double* query,
                                            const double& tolerance,
                                            const int& max_iterations,
                                            const bool aggressive_bounds,
                                            double* para_coord,
                                            double* phys_coord,
                                            double* phys_diff,
                                            double& distance,
                                            double& convergence_norm,
                                            double* first_derivatives,
                                            double* second_derivatives) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyVerboseProximity not implemented for",
      SplinepyWhatAmI());
}

void SplinepyBase::SplinepyElevateDegree(const int& para_dims) {
  splinepy::utils::PrintAndThrowError(
      "SplinepyElevateDegree not implemented for",
      SplinepyWhatAmI());
}

bool SplinepyBase::SplinepyReduceDegree(const int& para_dims,
                                        const double& tolerance) {
  splinepy::utils::PrintAndThrowError(
      "SplinepyReduceDegree not implemented for",
      SplinepyWhatAmI());
  return false;
}

bool SplinepyBase::SplinepyInsertKnot(const int& para_dim, const double& knot) {
  splinepy::utils::PrintAndThrowError("SplinepyInsertKnot not implemented for",
                                      SplinepyWhatAmI());
  return false;
}

bool SplinepyBase::SplinepyRemoveKnot(const int& para_dim,
                                      const double& knot,
                                      const double& tolerance) {
  splinepy::utils::PrintAndThrowError("SplinepyRemoveKnot not implemented for",
                                      SplinepyWhatAmI());
  return false;
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyMultiply(const std::shared_ptr<SplinepyBase>& a) const {
  splinepy::utils::PrintAndThrowError("SplinepyMultiply not implemented for",
                                      SplinepyWhatAmI());
  return std::shared_ptr<SplinepyBase>{};
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyAdd(const std::shared_ptr<SplinepyBase>& a) const {
  splinepy::utils::PrintAndThrowError("SplinepyAdd not implemented for",
                                      SplinepyWhatAmI());
  return std::shared_ptr<SplinepyBase>{};
}

std::shared_ptr<SplinepyBase> SplinepyBase::SplinepyCompose(
    const std::shared_ptr<SplinepyBase>& inner_function) const {
  splinepy::utils::PrintAndThrowError("SplinepyCompose not implemented for",
                                      SplinepyWhatAmI());
  return std::shared_ptr<SplinepyBase>{};
}

std::vector<std::shared_ptr<SplinepyBase>>
SplinepyBase::SplinepyComposeSensitivities(
    const std::shared_ptr<SplinepyBase>& inner_function) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyComposeSensitivities not implemented for",
      SplinepyWhatAmI());
  return std::vector<std::shared_ptr<SplinepyBase>>{};
}

std::vector<std::shared_ptr<SplinepyBase>>
SplinepyBase::SplinepySplit(const int& para_dim, const double& location) const {
  splinepy::utils::PrintAndThrowError("SplinepySplit not implemented for",
                                      SplinepyWhatAmI());
  return {std::shared_ptr<SplinepyBase>{}};
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyDerivativeSpline(const int* orders) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyDerivativeSpline is not implemented for",
      SplinepyWhatAmI());
  return std::shared_ptr<SplinepyBase>{};
}

std::vector<std::shared_ptr<SplinepyBase>>
SplinepyBase::SplinepyExtractBezierPatches() const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyBezierPatchExtraction is not implemented for",
      SplinepyWhatAmI());
  return {std::shared_ptr<SplinepyBase>{}};
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyExtractBoundary(const int& boundary_id) {
  splinepy::utils::PrintAndThrowError(
      "SplinepyExtractBoundary is not implemented for",
      SplinepyWhatAmI());
  return {std::shared_ptr<SplinepyBase>{}};
}

std::shared_ptr<SplinepyBase>
SplinepyBase::SplinepyExtractDim(const int& phys_dim) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyExtractDim is not implemented for",
      SplinepyWhatAmI());
  return {std::shared_ptr<SplinepyBase>{}};
}

std::shared_ptr<SplinepyBase> SplinepyBase::SplinepyCompositionDerivative(
    const std::shared_ptr<SplinepyBase>& inner,
    const std::shared_ptr<SplinepyBase>& inner_derivative) const {
  splinepy::utils::PrintAndThrowError(
      "SplinepyCompositionDerivative is not implemented for",
      SplinepyWhatAmI());
  return {std::shared_ptr<SplinepyBase>{}};
}

} // namespace splinepy::splines
