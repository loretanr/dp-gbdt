// Copyright 2010 Jiang Chen. All Rights Reserved.
// Author: Jiang Chen <criver@gmail.com>

#include "split_algo.h"

#include <set>
#include <string>
#include <unordered_set>
#include <vector>

#include "gtest/gtest.h"
#include "src/base/base.h"
#include "src/data_store/column.h"
#include "src/proto/config.pb.h"
#include "src/proto/tree.pb.h"

namespace gbdt {

class FindSplitPointTest : public ::testing::Test {
 protected:
  void SetUp() {
    for (uint i = 0; i < samples_.size(); ++i) {
      uint index = samples_[i];
      total_ += w_(index) * gradient_data_vec_[index];
    }
  }

  vector<GradientData> gradient_data_vec_ = {
    {-1, 1}, {-2, 1}, {2, 1}, {-1, 1}, {-2, 1}, {3, 1}, {2, 1}, {3, 1}, {0, 1}, {-2, 1}};
  vector<float> weights_ = {0.1, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1, 0.2, 0.1, 0.1};
  FloatVector w_ = [this](int i) { return weights_[i]; };
  FloatVector constant_weight_ = [](int i) { return 1.0; };
  vector<uint> samples_ = {0, 2, 3, 4, 1, 5, 7, 6};
  GradientData total_;
};

TEST_F(FindSplitPointTest, FindFloatSplitPoint) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, 5, 3, 1, 7, 5, 7, 3, 5}));

  Split split;

  CHECK(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));
  EXPECT_FLOAT_EQ(4.0, split.float_split().threshold());
  EXPECT_FLOAT_EQ(4.0, split.gain());
}

TEST_F(FindSplitPointTest, FindFloatSplitPointWithMissingValues) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, 5, 3, 1, NAN, 5, NAN, 3, 5}));

  Split split;
  CHECK(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));
  EXPECT_FLOAT_EQ(4.0, split.float_split().threshold());
  EXPECT_TRUE(split.float_split().missing_to_right_child());
  EXPECT_FLOAT_EQ(4.0, split.gain());
}

TEST_F(FindSplitPointTest, FindFloatSplitPointWithMissingValues2) {
  // In this test case, missing value is placed on the left most side and the split
  // threshold is less than the lowest feature value.
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, NAN, 3, 1, NAN, NAN, NAN, 3, NAN}));

  Split split;
  CHECK(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));
  EXPECT_FLOAT_EQ(0.999, split.float_split().threshold());
  EXPECT_FALSE(split.float_split().missing_to_right_child());
  EXPECT_FLOAT_EQ(4.0, split.gain());
}

TEST_F(FindSplitPointTest, FindStringSplit) {
  auto feature = Column::CreateStringColumn(
      "foo", {"1", "3", "5", "3", "1", "7", "5", "7", "3", "5"});
  Split split;
  CHECK(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));
  EXPECT_EQ(2, split.cat_split().internal_categorical_index_size());
  set<uint> value_set(split.cat_split().internal_categorical_index().begin(),
                      split.cat_split().internal_categorical_index().end());
  EXPECT_EQ(set<uint>({1, 2}), value_set);
  EXPECT_FLOAT_EQ(4.0, split.gain());
}

TEST_F(FindSplitPointTest, FindStringSplitOutOfOrder) {
  // "1" and "5" have score -1, "3" and "7" have score 1.
  auto feature = Column::CreateStringColumn(
      "foo", {"1", "3", "5", "3", "1", "7", "5", "7", "3", "5"});
  vector<double> g = {-1, 1, -1, 1, -1, 1, -1, 1, 1, -1};
  for (int i = 0; i < g.size(); ++i) {
    gradient_data_vec_[i].g = g[i];
  }
  Split split;
  CHECK(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));
  EXPECT_EQ(2, split.cat_split().internal_categorical_index_size());
  set<uint> value_set(split.cat_split().internal_categorical_index().begin(),
                      split.cat_split().internal_categorical_index().end());
  EXPECT_EQ(set<uint>({1, 3}), value_set);
  EXPECT_FLOAT_EQ(1.7066667, split.gain());
}

TEST_F(FindSplitPointTest, NotEnoughSamplesFloat) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, 5, 3, 1, 7, 5, 7, 3, 5}));

  Split split;
  vector<uint> samples = {0};
  EXPECT_FALSE(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples, Config(), total_, &split));
  EXPECT_FLOAT_EQ(0.0, split.gain());
  EXPECT_FALSE(split.has_float_split());
}

TEST_F(FindSplitPointTest, NotEnoughSamplesString) {
  auto feature = Column::CreateStringColumn(
      "foo", {"1", "3", "5", "3", "1", "7", "5", "7", "3", "5"});

  Split split;
  vector<uint> samples = {0};
  EXPECT_FALSE(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples, Config(), total_, &split));
  EXPECT_FLOAT_EQ(0.0, split.gain());
  EXPECT_FALSE(split.has_cat_split());
}

TEST_F(FindSplitPointTest, MinHessian) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, 5, 3, 1, 7, 5, 7, 3, 5}));

  Split split;

  Config config;
  config.set_min_hessian(5);
  EXPECT_FALSE(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, config, total_, &split));
  EXPECT_FLOAT_EQ(0.0, split.gain());
}

TEST_F(FindSplitPointTest, ConstFloatFeature) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 1, 1, 1, 1, 1, 1, 1, 1, 1}));

  Split split;
  EXPECT_FALSE(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));

  EXPECT_FLOAT_EQ(0.0, split.gain());
  EXPECT_FALSE(split.has_float_split());
}

TEST_F(FindSplitPointTest, ConstStringFeature) {
  auto feature = Column::CreateStringColumn(
      "foo", vector<string>({"a", "a", "a", "a", "a", "a", "a", "a", "a", "a"}));

  Split split;
  EXPECT_FALSE(FindBestSplit(feature.get(), w_, &gradient_data_vec_, samples_, Config(), total_, &split));

  EXPECT_FLOAT_EQ(0.0, split.gain());
  EXPECT_FALSE(split.has_cat_split());
}

TEST_F(FindSplitPointTest, IrrelevantFeature) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "foo", vector<float>({1, 3, 1, 3, 1, 3, 1, 3, 1, 3}));
  vector<double> g = {1, 1, -1, -1, 1, 1, -1, -1, 1, 1};
  vector<uint> samples = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  vector<double> h = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  GradientData total;
  total.g = 2.0;
  total.h = 10.0;
  Split split;
  EXPECT_FALSE(FindBestSplit(feature.get(), constant_weight_, &gradient_data_vec_, samples,
                             Config(), total, &split));

  EXPECT_FLOAT_EQ(0.0, split.gain());
  EXPECT_FALSE(split.has_float_split());
}

class PartitionTest : public ::testing::Test {
 protected:
  void SetUp() {
    feature0_ = Column::CreateStringColumn(
        "foo", { "red", "red", "blue", "green", "red", "green", "yellow", "green" });
    feature1_ = Column::CreateBucketizedFloatColumn(
        "bar", {5, 3, 7, 1, 2, 0, 6, 4});
    samples_.resize(feature0_->size());
    for (uint i = 0; i < samples_.size(); ++i) {
      samples_[i] = i;
    }
  }

  set<uint> SliceToSet(VectorSlice<uint> slice) {
    return set<uint>(slice.begin(), slice.end());
  }

  vector<uint> samples_;
  unique_ptr<Column> feature0_;
  unique_ptr<Column> feature1_;
};

TEST_F(PartitionTest, PartitionStringColumn) {
  Split split;
  // Red and green.
  split.mutable_cat_split()->add_internal_categorical_index(1);
  split.mutable_cat_split()->add_internal_categorical_index(3);
  auto slices = Partition(feature0_.get(), split, samples_);
  EXPECT_EQ(set<uint>({ 0, 1, 3, 4, 5, 7}), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({ 6, 2}), SliceToSet(slices.second));
}

TEST_F(PartitionTest, PartitionStringColumnString) {
  Split split;
  // Red and green.
  split.mutable_cat_split()->add_category("red");
  split.mutable_cat_split()->add_category("green");
  auto slices = Partition(feature0_.get(), split, samples_);
  EXPECT_EQ(set<uint>({ 0, 1, 3, 4, 5, 7}), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({ 6, 2}), SliceToSet(slices.second));
}


TEST_F(PartitionTest, PartitionStringColumnPartial) {
  Split split;
  // Red and green.
  split.mutable_cat_split()->add_internal_categorical_index(1);
  split.mutable_cat_split()->add_internal_categorical_index(3);
  auto slices = Partition(feature0_.get(), split, VectorSlice<uint>(samples_, 2, 4));
  EXPECT_EQ(3, slices.first.size());
  EXPECT_EQ(set<uint>({ 3, 4, 5}), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({ 2 }), SliceToSet(slices.second));
}

TEST_F(PartitionTest, PartitionFloatColumn) {
  Split split;
  split.mutable_float_split()->set_threshold(3.5);
  auto slices = Partition(feature1_.get(), split, samples_);
  EXPECT_EQ(set<uint>({1, 3, 4, 5 }), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({2, 0, 6, 7}), SliceToSet(slices.second));
}

TEST_F(PartitionTest, PartitionFloatColumnPartial) {
  Split split;
  split.mutable_float_split()->set_threshold(3.5);
  auto slices = Partition(feature1_.get(), split, VectorSlice<uint>(samples_, 2, 4));
  EXPECT_EQ(set<uint>({3, 4, 5 }), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({2}), SliceToSet(slices.second));
}

TEST_F(PartitionTest, PartitionFloatColumnWithMissing) {
  auto feature = Column::CreateBucketizedFloatColumn(
      "hello", {4, 5, NAN, 3, NAN, NAN, 2, 1});
  Split split;
  split.mutable_float_split()->set_threshold(2.5);
  // Missing to the left.
  auto slices = Partition(feature.get(), split, samples_);
  EXPECT_EQ(set<uint>({ 2, 4, 5, 6, 7 }), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({0, 3, 1}), SliceToSet(slices.second));

  // Missing to the right.
  split.mutable_float_split()->set_missing_to_right_child(true);
  slices = Partition(feature.get(), split, samples_);
  EXPECT_EQ(set<uint>({ 6, 7 }), SliceToSet(slices.first));
  EXPECT_EQ(set<uint>({0, 3, 1, 2, 4, 5}), SliceToSet(slices.second));
}

}  // namespace gbdt
