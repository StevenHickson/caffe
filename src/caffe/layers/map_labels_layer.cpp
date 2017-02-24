#include <algorithm>
#include <cfloat>
#include <vector>
#include <sstream>
#include <iostream>

#include "caffe/layers/map_labels_layer.hpp"
#include "caffe/util/math_functions.hpp"

void GetMappingFromFile(std::string mapping_file, std::vector< std::vector<int> > *coarse_mapping, std::vector<int> *fine_mapping) {
	//First let's get the coarse to fine mapping
	std::ifstream infile(mapping_file.c_str());
	std::string line;
	int line_num = 0;
	int max_fine = 0;
	while (std::getline(infile, line))
	{
		coarse_mapping->push_back(std::vector<int>());
		std::stringstream ss(line);
		int i;
		while (ss >> i)
		{
			if (i > max_fine)
				max_fine = i;
			(*coarse_mapping)[line_num].push_back(i);
			if (ss.peek() == ',')
				ss.ignore();
		}
		++line_num;
	}
	infile.close();

	//Not let's get the fine to coarse mapping
	fine_mapping->resize(max_fine + 1);
	for (int coarse = 0; coarse < coarse_mapping->size(); coarse++) {
		std::vector<int>::const_iterator p = (*coarse_mapping)[coarse].begin();
		while (p != (*coarse_mapping)[coarse].end())
			(*fine_mapping)[*p++] = coarse;
	}
}

namespace caffe {

template <typename Dtype>
void MapLabelsLayer<Dtype>::LayerSetUp(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  std::string mapping_file =
    this->layer_param_.map_labels_param().mapping_file();
  CHECK_NE(mapping_file, "");
  CHECK_GT(mapping_file.length(), 0);

  // Let's get the mapping
  GetMappingFromFile(mapping_file, &coarse_to_fine_mapping_, &fine_to_coarse_mapping_);
}

template <typename Dtype>
void MapLabelsLayer<Dtype>::Reshape(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
	top[0]->ReshapeLike(*(bottom[0]));
}

template <typename Dtype>
void MapLabelsLayer<Dtype>::Forward_cpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  
	const Dtype* old_label = bottom[0]->cpu_data();
	Dtype* new_label = top[0]->mutable_cpu_data();
	const int num = bottom[0]->num() * bottom[0]->channels() * bottom[0]->width() * bottom[0]->height();

	for (int i = 0; i < num; i++) {
		new_label[i] = fine_to_coarse_mapping_[old_label[i]];
	}
}

template <typename Dtype>
void MapLabelsLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
	const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
}

#ifdef CPU_ONLY
STUB_GPU(MapLabelsLayer);
#endif

INSTANTIATE_CLASS(MapLabelsLayer);
REGISTER_LAYER_CLASS(MapLabels);

}  // namespace caffe
