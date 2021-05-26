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

#include "loss_func_lambdamart.h"

#include <memory>
#include <vector>

#include "gtest/gtest.h"

#include "src/data_store/data_store.h"

namespace gbdt {

class PairwiseTest : public ::testing::Test {
 protected:
  void SetUp() {
    data_store_.Add(Column::CreateStringColumn("group0", {"1", "1", "1", "1"}));

    config_.set_pair_sampling_rate(kSamplingRate_);
  }

  void ExpectGradientEqual(const vector<GradientData>& expected,
                           const vector<GradientData>& gradient_data_vec) {
    for (int i = 0; i < expected.size(); ++i) {
      double avg_g = gradient_data_vec[i].g / kSamplingRate_;
      double avg_h = gradient_data_vec[i].h / kSamplingRate_;
      EXPECT_LT(fabs(expected[i].g - avg_g), 1e-3)
          << " at " << i << " actual " << avg_g << " vs " << expected[i].g;
      EXPECT_LT(fabs(expected[i].h - avg_h), 1e-4)
          << " at " << i << " actual " << avg_h << " vs " << expected[i].h;
    }
  }

  DataStore data_store_;
  const int kSamplingRate_ = 100000;
  Config config_;
  FloatVector w_ = [](int) { return 1.0; };
  FloatVector y_ = [](int i) { return i; };
};

TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessians) {
  vector<double> f = { 1, 0, 3, 2};
  vector<GradientData> gradient_data_vec;
  double c;

  unique_ptr<Pairwise> lambdamart(new LambdaMART(config_));
  lambdamart->Init(data_store_.num_rows(), w_, y_, data_store_.GetStringColumn("group0"));
  lambdamart->ComputeFunctionalGradientsAndHessians(f, &c, &gradient_data_vec, nullptr);
  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);

  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-0.092, 0.131}, {-0.00817, 0.0543}, {-0.04, 0.1358 }, {0.141, 0.128} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

}  // namespace gbdt
