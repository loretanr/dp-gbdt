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

#include "group.h"

#include <algorithm>
#include <random>

#include "src/data_store/column.h"

namespace gbdt {

Group::Group(vector<uint>&& group, FloatVector y) : group_(group), y_(y) {
  // Sort groups on the descendinxxgxg order of y.
  sort(group_.begin(), group_.end(),
       [&] (uint i, uint j) { return y(i) > y(j); });
  // Rank is initialize to identity order.
  ranks_.resize(group_.size());
  for (int i = 0; i < ranks_.size(); ++i) {
    ranks_[i] = i;
  }

  // Find change boundaries of targets.
  int last_boundary = 0;
  for (int i = 1; i < group_.size(); ++i) {
    if (y(group_[i - 1]) != y(group_[i])) {
      // The pairs consists of this target group and all items following it.
      uint block_size = (i - last_boundary);
      num_pairs_ += block_size * (group_.size() - i);
      pair_map_.insert(make_pair(num_pairs_, make_pair(block_size, i)));
      last_boundary = i;
    }
  }
}

void Group::Rerank(const vector<double>& f) {
  // Computes the ranking based on the current function.
  vector<uint> ranking(size());
  for (int i = 0; i < ranking.size(); ++i) {
    ranking[i] = i;
  }

  // Sort by f.
  sort(ranking.begin(), ranking.end(),
       [&group=group_, &f](uint i, uint j) { return f[group[i]] > f[group[j]]; });

  for (int i = 0; i < ranking.size(); ++i) {
    ranks_[ranking[i]] = i;
  }
}

// TODO(criver): describe the algorithm.
pair<uint, uint> Group::SamplePair(std::mt19937* generator) const {
  std::uniform_int_distribution<uint64> sampler(0, num_pairs_ - 1);
  uint64 pair_index = sampler(*generator);
  // Given a pair_index, try to find the actual pair.
  auto it = pair_map_.lower_bound(pair_index + 1);
  int block_size = it->second.first;
  uint64 start_of_neg = it->second.second;
  uint64 local_pair_index = it->first - pair_index - 1;
  uint pos_index = start_of_neg - 1 - local_pair_index % block_size;
  uint neg_index = start_of_neg + local_pair_index / block_size;
  return make_pair(pos_index, neg_index);
}

}  // namespace gbdt
