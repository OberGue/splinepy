#include "BSplineLib/Splines/b_spline.hpp"
template class bsplinelib::splines::BSpline<1>;
template class bsplinelib::splines::BSpline<2>;
template class bsplinelib::splines::BSpline<3>;
#ifdef SPLINEPY_MORE
template class bsplinelib::splines::BSpline<4>;
template class bsplinelib::splines::BSpline<5>;
template class bsplinelib::splines::BSpline<6>;
template class bsplinelib::splines::BSpline<7>;
template class bsplinelib::splines::BSpline<8>;
template class bsplinelib::splines::BSpline<9>;
template class bsplinelib::splines::BSpline<10>;
#endif
