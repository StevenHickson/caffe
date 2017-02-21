// --------------------------------------------------------
// Multitask Network Cascade
// Modified from caffe-fast-rcnn (https://github.com/rbgirshick/caffe-fast-rcnn)
// Copyright (c) 2016, Haozhi Qi
// Licensed under The MIT License [see LICENSE for details]
// --------------------------------------------------------

#ifndef CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_
#define CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class SmoothL1LossLayer : public LossLayer<Dtype> {
 public:
  explicit SmoothL1LossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "SmoothL1Loss"; }

  virtual inline int ExactNumBottomBlobs() const { return -1; }
  virtual inline int MinBottomBlobs() const { return 2; }
  virtual inline int MaxBottomBlobs() const { return 4; }

  /**
   * Unlike most loss layers, in the SmoothL1LossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> errors_;
  Blob<Dtype> ones_;
  bool has_weights_;
  Dtype sigma2_;
};

}  // namespace caffe

#endif  // CAFFE_SMOOTH_L1_LOSS_LAYER_HPP_
