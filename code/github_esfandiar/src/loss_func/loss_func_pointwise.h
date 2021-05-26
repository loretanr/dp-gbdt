/* Copyright 2016 Jiang Chen <criver@gmail.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef LOSS_FUNC_POINTWISE_H_
#define LOSS_FUNC_POINTWISE_H_

#include <functional>
#include <string>
#include <vector>

#include "loss_func.h"
#include "loss_func_math.h"

namespace gbdt {

// Base class for pointwise loss function. A point_wise loss function need to
// compute the loss, negative gradient and hessian.
class Pointwise : public LossFunc {
public:
  Pointwise(PointwiseLossFunc loss_func);
  virtual Status Init(int num_rows, FloatVector w, FloatVector y, const StringColumn* unused_group_column) override;
  virtual void ComputeFunctionalGradientsAndHessians(const vector<double>& f,
                                                     double* c,
                                                     vector<GradientData>* gradient_data_vec,
                                                     string* progress) override;

private:
  string PrepareProgressMessage(double loss);

  PointwiseLossFunc loss_func_;
  FloatVector w_;
  FloatVector y_;
  double initial_loss_ = -1;
  double weight_sum_ = 0;
  // Division of [1, sample_size] into slices to help multithreading.
  vector<pair<uint, uint>> slices_;
};

}  // namespace gbdt

#endif  // LOSS_FUNC_POINTWISE_H_
