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

#include "loss_func_pairwise.h"

#include <functional>
#include <numeric>
#include <vector>

#include "gtest/gtest.h"

#include "loss_func_math.h"
#include "loss_func_pairwise_logloss.h"
#include "src/data_store/data_store.h"

namespace gbdt {

class PairwiseTest : public ::testing::Test {
 protected:
  void SetUp() {
    data_store_.Add(Column::CreateStringColumn("group0", {"1", "1", "1", "1"}));
    data_store_.Add(Column::CreateStringColumn("group1", {"0", "0", "1", "1"}));

    config_.set_pair_sampling_rate(kSamplingRate_);
  }

  void ExpectGradientEqual(const vector<GradientData>& expected,
                           const vector<GradientData>& gradient_data_vec) {
    for (int i = 0; i < expected.size(); ++i) {
      double avg_g = gradient_data_vec[i].g / kSamplingRate_;
      double avg_h = gradient_data_vec[i].h / kSamplingRate_;
      EXPECT_LT(fabs(expected[i].g - avg_g), 5e-2)
          << " at " << i << " actual " << avg_g << " vs " << expected[i].g;
      EXPECT_LT(fabs(expected[i].h - avg_h), 5e-2)
          << " at " << i << " actual " << avg_h << " vs " << expected[i].h;
    }
  }

  unique_ptr<Pairwise> CreateAndInitPairwiseLoss(const string& group_name) {
    unique_ptr<Pairwise> pairwise(new PairwiseLogLoss(config_));

    pairwise->Init(data_store_.num_rows(), w_, y_, data_store_.GetStringColumn(group_name));
    return pairwise;
  }

  DataStore data_store_;
  FloatVector w_ = [](int) { return 1.0; };
  FloatVector y_ = [](int i) { return i; };
  vector<double> f_ = { 0, 0, 0, 0};
  // Set sampleing_rate to 10000 so that g and h are more stable.
  const int kSamplingRate_ = 100000;
  Config config_;
};

// All instances are in one group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansOneGroup) {
  vector<GradientData> gradient_data_vec;
  double c;
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss("group0");
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-0.5, 0.5}, {-0.5/3, 0.5}, {0.5/3, 0.5}, {0.5, 0.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

// When on group_columnis specified, every instance is put in one group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansNoGroup) {
  // When no group specified, every instance is put in one group.
  vector<GradientData> gradient_data_vec;
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss("");
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // The gradients reflect the relative order of the original targets.
  vector<GradientData> expected = { {-0.5, 0.5}, {-0.5/3, 0.5}, {0.5/3, 0.5}, {0.5, 0.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

// All instances are in two group.
TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansTwoGroups) {
  vector<GradientData> gradient_data_vec;

  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss("group1");
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // Because of the grouping, the gradients of the two groups {0, 1} and {2, 3}
  // are similar in the magnitude since the targets are not compared
  // across groups.
  vector<GradientData> expected = { {-0.5, 0.5}, {0.5, 0.5}, {-0.5, 0.5}, {0.5, 0.5} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

TEST_F(PairwiseTest, TestComputeFunctionalGradientsAndHessiansWeightByDeltaTarget) {
  vector<GradientData> gradient_data_vec;

  config_.set_pair_weight_by_delta_target(true);
  unique_ptr<Pairwise> pairwise = CreateAndInitPairwiseLoss("group0");
  double c;
  pairwise->ComputeFunctionalGradientsAndHessians(f_, &c, &gradient_data_vec, nullptr);

  // c is zero for all pairwise losses.
  EXPECT_FLOAT_EQ(0, c);
  // More weights are put on higher target separation.
  vector<GradientData> expected = { {-1, 1}, {-1.0/3, 2.0/3}, {1.0/3, 2.0/3}, {1, 1} };
  ExpectGradientEqual(expected, gradient_data_vec);
}

}  // namespace gbdt
