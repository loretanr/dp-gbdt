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

#include "tsv_block.h"

#include "gtest/gtest.h"

namespace gbdt {

TEST(TSVBlockTest, TestHeaderlessBlock) {
  TSVBlock tsv_block(
      "src/data_store/testdata/tsv_data_store_test/block-0.tsv",
      {0, 3}, {1, 4}, false);
  ASSERT_EQ(2, tsv_block.float_columns().size());
  ASSERT_EQ(2, tsv_block.string_columns().size());
  EXPECT_EQ(vector<float>({0, 1, 2}), tsv_block.float_columns()[0]);
  EXPECT_EQ(vector<float>({5.4, 4.3, 3.2}), tsv_block.float_columns()[1]);
  EXPECT_EQ(vector<string>({"red", "blue", "green"}), tsv_block.string_columns()[0]);
  EXPECT_EQ(vector<string>({"rainy", "clear", "cloudy"}), tsv_block.string_columns()[1]);
}

TEST(TSVBlockTest, TestHeaderedBlock) {
  TSVBlock tsv_block(
      "src/data_store/testdata/tsv_data_store_test/block-0-with-header.tsv",
      {0, 3}, {1, 4}, true);
  ASSERT_EQ(2, tsv_block.float_columns().size());
  ASSERT_EQ(2, tsv_block.string_columns().size());
  EXPECT_EQ(vector<float>({0, 1, 2}), tsv_block.float_columns()[0]);
  EXPECT_EQ(vector<float>({5.4, 4.3, 3.2}), tsv_block.float_columns()[1]);
  EXPECT_EQ(vector<string>({"red", "blue", "green"}), tsv_block.string_columns()[0]);
  EXPECT_EQ(vector<string>({"rainy", "clear", "cloudy"}), tsv_block.string_columns()[1]);
}

TEST(TSVBlockTest, TestBlockWithMissingValue) {
  TSVBlock tsv_block(
      "src/data_store/testdata/tsv_data_store_test/data_with_missing.tsv",
      {0, 1}, {}, true);
  ASSERT_EQ(2, tsv_block.float_columns().size());
  // First column.
  EXPECT_FLOAT_EQ(32.33333342, tsv_block.float_columns()[0][0]);
  EXPECT_TRUE(isnan(tsv_block.float_columns()[0][1]));
  EXPECT_TRUE(isnan(tsv_block.float_columns()[0][2]));
  EXPECT_FLOAT_EQ(-593.323, tsv_block.float_columns()[0][3]);
  EXPECT_TRUE(isnan(tsv_block.float_columns()[0][4]));
  // Second column.
  EXPECT_FLOAT_EQ(-3.295, tsv_block.float_columns()[1][0]);
  EXPECT_FLOAT_EQ(95231.405, tsv_block.float_columns()[1][1]);
  EXPECT_FLOAT_EQ(32.11392, tsv_block.float_columns()[1][2]);
  EXPECT_TRUE(isnan(tsv_block.float_columns()[1][3]));
  EXPECT_FLOAT_EQ(0.33234, tsv_block.float_columns()[1][4]);
}

}  // namespace gbdt
