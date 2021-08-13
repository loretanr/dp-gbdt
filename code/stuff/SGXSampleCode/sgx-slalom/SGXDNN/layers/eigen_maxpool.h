
// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2014 Benoit Steiner <benoit.steiner.goog@gmail.com>
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.
#ifndef EIGEN_CXX11_NEURAL_NETWORKS_POOLING_H
#define EIGEN_CXX11_NEURAL_NETWORKS_POOLING_H

#include <unsupported/Eigen/CXX11/Tensor>

using namespace std;

namespace Eigen {

/** SpatialMaxPooling
  * \ingroup CXX11_NeuralNetworks_Module
  *
  * \brief Applies a max-pooling over a multichannel input image.
  *
  * The input parameter is expected to be a with a rank of 4 (channels, height, width, others in col-major, and the reverse of that in row-major).
  *
  * The result can be assigned to a tensor of rank equal to the rank of the input. The dimensions of the result will be channels, height, width, and others (in col-major, and the reverse of that if the input was row-major).
  *
  * The order of the width and height dimensions can be swapped if needed.
  *
*/
#if !defined(EIGEN_HAS_INDEX_LIST)
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::MaxReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, const Eigen::array<int, 2>, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#else
template <typename Input>
EIGEN_ALWAYS_INLINE
static const TensorReshapingOp<const Eigen::DSizes<typename internal::traits<Input>::Index, internal::traits<Input>::NumDimensions>, const TensorReductionOp<internal::MaxReducer<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>, typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type, const TensorImagePatchOp<Dynamic, Dynamic, const Input> > >
#endif
SpatialMaxPooling(const Input& input, DenseIndex patchRows, DenseIndex patchCols,
                  DenseIndex strideRows, DenseIndex strideCols, const PaddingType padding_type,
                  DenseIndex in_strideRows = 1, DenseIndex in_strideCols = 1)
{
  EIGEN_STATIC_ASSERT(internal::traits<Input>::NumDimensions == 4, YOU_MADE_A_PROGRAMMING_MISTAKE);

  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar, internal::traits<Input>::NumDimensions, internal::traits<Input>::Layout, TensorIndex> > in(input);

  const DenseIndex patchRowsEff = patchRows + (patchRows - 1) * (in_strideRows - 1);
  const DenseIndex patchColsEff = patchCols + (patchCols - 1) * (in_strideCols - 1);

  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);
  static const int idxRows = isColMajor ? 1 : 2;
  static const int idxCols = isColMajor ? 2 : 1;

  // Molds the output of the reduction into the shape expected by the user.
  // (assuming col-major):
  // - 1st dim: channels
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  Eigen::DSizes<TensorIndex, internal::traits<Input>::NumDimensions> post_reduce_dims;
  post_reduce_dims[0] = in.dimension(0);
  if (padding_type == PADDING_VALID) {
    post_reduce_dims[idxRows] = numext::ceil((in.dimension(idxRows) - patchRowsEff + 1.f) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil((in.dimension(idxCols) - patchColsEff + 1.f) / static_cast<float>(strideCols));
  } else {
    post_reduce_dims[idxRows] = numext::ceil(in.dimension(idxRows) / static_cast<float>(strideRows));
    post_reduce_dims[idxCols] = numext::ceil(in.dimension(idxCols) / static_cast<float>(strideCols));
  }
  post_reduce_dims[3] = in.dimension(3);

#if !defined(EIGEN_HAS_INDEX_LIST)
  // nvcc doesn't support cxx11
  Eigen::array<int, 2> reduction_dims;
  if (isColMajor) {
    reduction_dims[0] = 1;
    reduction_dims[1] = 2;
  } else {
    reduction_dims[0] = 2;
    reduction_dims[1] = 3;
  }
#else
  // Take advantage of cxx11 to give the compiler information it can use to
  // optimize the code.
  typename internal::conditional<internal::traits<Input>::Layout == ColMajor, const Eigen::IndexList<Eigen::type2index<1>, Eigen::type2index<2> >, const Eigen::IndexList<Eigen::type2index<2>, Eigen::type2index<3> > >::type reduction_dims;
#endif

  return input.extract_image_patches(patchRows, patchCols, strideRows, strideCols, in_strideRows, in_strideCols, padding_type, -Eigen::NumTraits<typename internal::remove_const<typename internal::traits<Input>::Scalar>::type>::highest()).maximum(reduction_dims).reshape(post_reduce_dims);
}

} // end namespace Eigen

#endif // EIGEN_CXX11_NEURAL_NETWORKS_POOLING_H
