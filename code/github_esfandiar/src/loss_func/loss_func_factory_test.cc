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

#include "loss_func_factory.h"

#include "gtest/gtest.h"
#include "src/proto/config.pb.h"

namespace gbdt {

TEST(LossFuncFactoryTest, TestLossFuncCreation) {
  Config config;

  EXPECT_EQ(nullptr, LossFuncFactory::CreateLossFunc(config));

  config.set_loss_func("mse");
  EXPECT_NE(nullptr, LossFuncFactory::CreateLossFunc(config));

  config.set_loss_func("logloss");
  EXPECT_NE(nullptr, LossFuncFactory::CreateLossFunc(config));

  config.set_loss_func("huberized_hinge");
  EXPECT_NE(nullptr, LossFuncFactory::CreateLossFunc(config));
}

}  // namespace gbdt
