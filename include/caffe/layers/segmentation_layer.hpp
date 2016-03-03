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
    virtual inline int MinBottomBlobs() const { return 1; }
    virtual inline int MinTopBlobs() const { return 1; }
    virtual inline int MaxBottomBlobs() const { return 2; }
    virtual inline int MaxTopBlobs() const { return 3; }

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

    int Segmentation(std::vector<cv::Mat> &in, cv::Mat &out, const int segStartNumber);

    //Generic segmentation variables
    int num_segments_;
    int bbox_extension_;
    int method_;

    //Method = 0 = FH
    float smoothing_;
    float k_;
    int min_size_;

    //Method = 1 = SLIC
    int s_;
    int m_;
    int iter_;
  };

}  // namespace caffe

#endif  // CAFFE_SEGMENTATION_LAYER_HPP_
