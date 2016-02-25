#ifndef CAFFE_SEGMENTATION_LAYER_HPP_
#define CAFFE_SEGMENTATION_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {

  /**
  * @brief Segment the image into a batch of segments
  *
  * TODO(dox): thorough documentation for Forward, Backward, and proto params.
  */
  template <typename Dtype>
  class SegmentationLayer : public Layer<Dtype> {
  public:
    explicit SegmentationLayer(const LayerParameter& param)
      : Layer<Dtype>(param) {}

    virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Segmentation"; }
    virtual inline int ExactNumBottomBlobs() const { return 2; }
    virtual inline int MinTopBlobs() const { return 1; }

  protected:
    virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
    /*virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);*/
    virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

    }
    /*virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
    }*/

    int height_;
    int width_;
    int num_segments_;
    //int data_height_;
    //int data_width_;
    int seg_parameter_;
  };

}  // namespace caffe

#endif  // CAFFE_SEGMENTATION_LAYER_HPP_
