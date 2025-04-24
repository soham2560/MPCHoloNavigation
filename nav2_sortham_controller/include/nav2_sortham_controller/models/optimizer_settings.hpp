// Copyright (c) 2022 Samsung Research America, @artofnothingness Alexey Budyakov
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef NAV2_SORTHAM_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
#define NAV2_SORTHAM_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_

#include <cstddef>
#include "nav2_sortham_controller/models/constraints.hpp"

namespace sortham::models
{

/**
 * @struct sortham::models::OptimizerSettings
 * @brief Settings for the optimizer to use
 */
struct OptimizerSettings
{
    models::ControlConstraints base_constraints{0, 0, 0, 0};
    models::ControlConstraints constraints{0, 0, 0, 0};
    models::SamplingStd sampling_std{0, 0, 0};
  float model_dt{0};
  float temperature{0};
  float gamma{0};
  unsigned int batch_size{0};
  unsigned int time_steps{0};
  unsigned int iteration_count{0};
    bool shift_control_sequence{false};
    size_t retry_attempt_limit{0};

  
    double weight_x = 50.0;
    double weight_y = 50.0;
    double weight_theta = 10.0;
    double weight_v = 1.0;
    double weight_w = 1.5;
    double weight_vy = 1.0;
    double weight_v_accel = 0.5;
    double weight_w_accel = 0.8;
    double weight_vy_accel = 0.5;
    double weight_terminal_x = -1.0;
    double weight_terminal_y = -1.0;
    double weight_terminal_theta = -1.0;

    double max_v_change = 0.1;
    double max_w_change = 0.5;
    double max_vy_change = 0.1;

    bool include_obstacle_constraints = false;
    double robot_radius = 0.25;
    double inflation_radius = 0.1;
};

}  // namespace sortham::models

#endif  // NAV2_SORTHAM_CONTROLLER__MODELS__OPTIMIZER_SETTINGS_HPP_
