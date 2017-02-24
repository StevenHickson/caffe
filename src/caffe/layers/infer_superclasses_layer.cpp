#include <algorithm>
#include <cfloat>
#include <vector>
#include <sstream>
#include <iostream>

#include "caffe/layers/infer_superclasses_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

	template <typename Dtype>
	void InferSuperclassesLayer<Dtype>::LayerSetUp(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		MapLabelsLayer<Dtype>::LayerSetUp(bottom, top);
	}

	template <typename Dtype>
	void InferSuperclassesLayer<Dtype>::Reshape(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
		//top[0]->ReshapeLike(*(bottom[0]));
		vector<int> shape = bottom[0]->shape();
		shape[1] = MapLabelsLayer<Dtype>::coarse_to_fine_mapping_.size();
		top[0]->Reshape(shape);
	}

	template <typename Dtype>
	void InferSuperclassesLayer<Dtype>::Forward_cpu(
		const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

		const Dtype* fine_preds = bottom[0]->cpu_data();
		Dtype* coarse_preds = top[0]->mutable_cpu_data();
		//const int num = bottom[0]->num() * bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();
		vector<int> shape = bottom[0]->shape();

		//Init to zero
		caffe_set(top[0]->count(), Dtype(0), coarse_preds);

		// For every fully connected output, sum the values and set the equivalent coarse value equal to that
		//There has definitely got to be a better way to do this with selective channel adding.
		//for (int n = 0; n < shape[0]; n++) {
		//	for (int c = 0; c < shape[1]; c++) {
		//		for (int h = 0; h < shape[2]; h++) {
		//			for (int w = 0; w < shape[3]; w++) {
		//				//caffe_set(1, )
		//				*(coarse_preds + top[0]->offset(n,MapLabelsLayer<Dtype>::fine_to_coarse_mapping_[c],h,w)) += 
		//			}
		//		}
		//	}
		//}

		// For every coarse channel, extract the fine channels and sum them to get the new coarse channel
		const int num = shape[2] * shape[3];
		for (int coarse = 0; coarse < MapLabelsLayer<Dtype>::coarse_to_fine_mapping_.size(); coarse++) {
			std::vector<int>::const_iterator p = MapLabelsLayer<Dtype>::coarse_to_fine_mapping_[coarse].begin();
			while (p != MapLabelsLayer<Dtype>::coarse_to_fine_mapping_[coarse].end()) {
				for (int n = 0; n < shape[0]; n++) {
					//For every fine channel, sum the coarse channel
					Dtype *q = coarse_preds + top[0]->offset(n, coarse, 0, 0);
					caffe_add(num, fine_preds + bottom[0]->offset(n, *p, 0, 0), q, q);
				}
				p++;
			}
		}
	}

	template <typename Dtype>
	void InferSuperclassesLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
		const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

		const Dtype* top_diff = top[0]->cpu_diff();
		Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();

		vector<int> shape = top[0]->shape();
		int num = shape[2] * shape[3];
		if (propagate_down[0]) {
			//For every fine channel, retrieve the coarse channel number and set the fine diff = to that diff
			for (int fine = 0; fine < MapLabelsLayer<Dtype>::fine_to_coarse_mapping_.size(); fine++) {
				for (int n = 0; n < shape[0]; n++) {
					//For every fine channel, set the fine_diff to the coarse diff
					caffe_copy(num, top_diff + top[0]->offset(n, MapLabelsLayer<Dtype>::fine_to_coarse_mapping_[fine], 0, 0), bottom_diff + bottom[0]->offset(n, fine, 0, 0));
				}
			}
		}
	}

#ifdef CPU_ONLY
	STUB_GPU(InferSuperclassesLayer);
#endif

	INSTANTIATE_CLASS(InferSuperclassesLayer);
	REGISTER_LAYER_CLASS(InferSuperclasses);

}  // namespace caffe
