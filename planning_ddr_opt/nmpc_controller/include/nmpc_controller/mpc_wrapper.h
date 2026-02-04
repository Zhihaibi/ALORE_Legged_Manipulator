/*    rpg_quadrotor_mpc
 *    A model predictive control implementation for quadrotors.
 *    Copyright (C) 2017-2018 Philipp Foehn, 
 *    Robotics and Perception Group, University of Zurich
 * 
 *    Intended to be used with rpg_quadrotor_control and rpg_quadrotor_common.
 *    https://github.com/uzh-rpg/rpg_quadrotor_control
 *
 *    This program is free software: you can redistribute it and/or modify
 *    it under the terms of the GNU General Public License as published by
 *    the Free Software Foundation, either version 3 of the License, or
 *    (at your option) any later version.
 *
 *    This program is distributed in the hope that it will be useful,
 *    but WITHOUT ANY WARRANTY; without even the implied warranty of
 *    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *    GNU General Public License for more details.
 *
 *    You should have received a copy of the GNU General Public License
 *    along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */


#pragma once

#include <Eigen/Eigen>
#include <ros/ros.h>

namespace Tracked_nmpc {

#include "acado_auxiliary_functions.h"
#include "acado_common.h"

static constexpr int kSamples = ACADO_N;      // number of samples
static constexpr int kStateSize = ACADO_NX;   // number of states
static constexpr int kRefSize = ACADO_NY;     // number of reference states
static constexpr int kEndRefSize = ACADO_NYN; // number of end reference states
static constexpr int kInputSize = ACADO_NU;   // number of inputs
static constexpr int kCostSize = ACADO_NY - ACADO_NU; // number of state costs
static constexpr int kOdSize = ACADO_NOD;     // number of online data

class MpcWrapper
{
 public:

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MpcWrapper();
  MpcWrapper(
    const Eigen::Ref<const Eigen::Matrix<double, kCostSize, kCostSize>> Q,
    const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kInputSize>> R);

  bool setCosts(
    const Eigen::Ref<const Eigen::Matrix<double, kCostSize, kCostSize>> Q,
    const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kInputSize>> R,
    const double state_cost_scaling = 0.0, const double input_cost_scaling = 0.0);

  bool setLimits(double min_thrust, double max_thrust,
    double max_rollpitchrate, double max_yawrate);
  // bool setCameraParameters(
  //   const Eigen::Ref<const Eigen::Matrix<double, 3, 1>>& p_B_C,
  //   Eigen::Quaternion<double>& q_B_C);

  // xv yr yl
  bool setICRParameters(
    const Eigen::Ref<const Eigen::Matrix<double, 3, 1>>& p_B_I);
  bool setPointOfInterest(
    const Eigen::Ref<const Eigen::Matrix<double, 3, 1>>& position);

  bool setReferencePose(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);
  bool setTrajectory(
    const Eigen::Ref<const Eigen::Matrix<double, kStateSize, kSamples+1>> states,
    const Eigen::Ref<const Eigen::Matrix<double, kInputSize, kSamples+1>> inputs);

  bool solve(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state);
  bool update(const Eigen::Ref<const Eigen::Matrix<double, kStateSize, 1>> state,
              bool do_preparation = true);
  bool prepare();

  void getState(const int node_index,
    Eigen::Ref<Eigen::Matrix<double, kStateSize, 1>> return_state);
  void getStates(
    Eigen::Ref<Eigen::Matrix<double, kStateSize, kSamples+1>> return_states);
  void getInput(const int node_index,
    Eigen::Ref<Eigen::Matrix<double, kInputSize, 1>> return_input);
  void getInputs(
    Eigen::Ref<Eigen::Matrix<double, kInputSize, kSamples>> return_input);
  double getTimestep() { return dt_; }

 private:
  Eigen::Map<Eigen::Matrix<float, kRefSize, kSamples, Eigen::ColMajor>>
    acado_reference_states_{acadoVariables.y};

  Eigen::Map<Eigen::Matrix<float, kEndRefSize, 1, Eigen::ColMajor>>
    acado_reference_end_state_{acadoVariables.yN};

  Eigen::Map<Eigen::Matrix<float, kStateSize, 1, Eigen::ColMajor>>
    acado_initial_state_{acadoVariables.x0};

  Eigen::Map<Eigen::Matrix<float, kStateSize, kSamples+1, Eigen::ColMajor>>
    acado_states_{acadoVariables.x};

  Eigen::Map<Eigen::Matrix<float, kInputSize, kSamples, Eigen::ColMajor>>
    acado_inputs_{acadoVariables.u};

  Eigen::Map<Eigen::Matrix<float, kOdSize, kSamples+1, Eigen::ColMajor>>
    acado_online_data_{acadoVariables.od};

  Eigen::Map<Eigen::Matrix<float, kRefSize, kRefSize * kSamples>>
    acado_W_{acadoVariables.W};

  Eigen::Map<Eigen::Matrix<float, kEndRefSize, kEndRefSize>>
    acado_W_end_{acadoVariables.WN};

  Eigen::Map<Eigen::Matrix<float, 4, kSamples, Eigen::ColMajor>>
    acado_lower_bounds_{acadoVariables.lbValues};

  Eigen::Map<Eigen::Matrix<float, 4, kSamples, Eigen::ColMajor>>
    acado_upper_bounds_{acadoVariables.ubValues};

  Eigen::Matrix<double, kRefSize, kRefSize> W_ = Eigen::Matrix<double, kRefSize, kRefSize>::Zero();
  Eigen::Matrix<double, kEndRefSize, kEndRefSize> WN_ = Eigen::Matrix<double, kEndRefSize, kEndRefSize>::Zero();

  // Eigen::Matrix<double, kRefSize, kRefSize> W_ = (Eigen::Matrix<double, kRefSize, 1>() <<
  //   10 * Eigen::Matrix<double, 3, 1>::Ones(),
  //   100 * Eigen::Matrix<double, 4, 1>::Ones(),
  //   10 * Eigen::Matrix<double, 3, 1>::Ones(),
  //   Eigen::Matrix<double, 2, 1>::Zero(),
  //   1, 10, 10, 1).finished().asDiagonal();

  // Eigen::Matrix<double, kEndRefSize, kEndRefSize> WN_ =
  //   W_.block(0, 0, kEndRefSize, kEndRefSize);

  bool acado_is_prepared_{false};
  const double dt_{0.01};
  // const Eigen::Matrix<real_t, kInputSize, 1> kHoverInput_ =
  //   (Eigen::Matrix<real_t, kInputSize, 1>() << 9.81, 0.0, 0.0, 0.0).finished();
  const Eigen::Matrix<real_t, kInputSize, 1> kHoverInput_ = Eigen::Matrix<real_t, kInputSize, 1>::Zero();
};

};
