#include <stdint.h>
#include <algorithm>
#include <string>
#include <utility>
#include <vector>

#include "boost/scoped_ptr.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"
#include "caffe/layers/image_data_layer.hpp"

//For debug
//#include "opencv2/opencv.hpp"

using namespace caffe;  // NOLINT(build/namespaces)

using std::max;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
	"The backend {leveldb, lmdb} containing the images");
//DEFINE_string(filename, "train.txt", "The filename with the list of train images and labels");
DEFINE_bool(use_db, true, "Dictates whether to use a db {leveldb, lmdb} or a list of images and labels");
DEFINE_int32(height, 227, "The new height of images if resizing is needed");
DEFINE_int32(width, 227, "The new width of images if resizing is needed");
DEFINE_int32(channels, 3, "The number of channels");
DEFINE_int32(batch_size, 32, "The batch size");
DEFINE_int32(num, 1, "the total number of images");

/*void getMatFromBlobProto(BlobProto &blob, cv::Mat *img) {
	*img = cv::Mat(FLAGS_height, FLAGS_width, CV_32FC3);
	cv::Mat_<cv::Vec3f>::iterator p = img->begin<cv::Vec3f>();
	for (int t = 0; t < blob.data_size(); ++t)
		*p++ = cv::Vec3f(blob.data(t++), blob.data(t++), blob.data(t));
}*/

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the mean_image of a set of images given by"
		" a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_mean [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	/* Let's temporarily remove this since we need to rewrite it
	if (argc < 2 || argc > 3) {
	  gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_mean");
	  return 1;
	}*/

	BlobProto sum_blob;
	int count = 0;

	//For debug
	//cv::Mat sum_img, sum_img2, sum_img_final;

	if (FLAGS_use_db) {
		scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
		db->Open(argv[1], db::READ);
		scoped_ptr<db::Cursor> cursor(db->NewCursor());
		// load first datum
		Datum datum;
		datum.ParseFromString(cursor->value());

		if (DecodeDatumNative(&datum)) {
			LOG(INFO) << "Decoding Datum";
		}

		sum_blob.set_num(1);
		sum_blob.set_channels(datum.channels());
		sum_blob.set_height(datum.height());
		sum_blob.set_width(datum.width());
		const int data_size = datum.channels() * datum.height() * datum.width();
		int size_in_datum = std::max<int>(datum.data().size(),
			datum.float_data_size());
		for (int i = 0; i < size_in_datum; ++i) {
			sum_blob.add_data(0.);
		}
		LOG(INFO) << "Starting Iteration";
		while (cursor->valid()) {
			Datum datum;
			datum.ParseFromString(cursor->value());
			DecodeDatumNative(&datum);

			const std::string& data = datum.data();
			size_in_datum = std::max<int>(datum.data().size(),
				datum.float_data_size());
			CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
				size_in_datum;
			if (data.size() != 0) {
				CHECK_EQ(data.size(), size_in_datum);
				for (int i = 0; i < size_in_datum; ++i) {
					sum_blob.set_data(i, sum_blob.data(i) + (uint8_t)data[i]);
				}
			}
			else {
				CHECK_EQ(datum.float_data_size(), size_in_datum);
				for (int i = 0; i < size_in_datum; ++i) {
					sum_blob.set_data(i, sum_blob.data(i) +
						static_cast<float>(datum.float_data(i)));
				}
			}
			++count;
			if (count % 10000 == 0) {
				LOG(INFO) << "Processed " << count << " files.";
			}
			cursor->Next();
		}

		if (count % 10000 != 0) {
			LOG(INFO) << "Processed " << count << " files.";
		}
	}
	else {
		Blob<float>* const blob_top_data_ = new Blob<float>();
		Blob<float>* const blob_top_label_ = new Blob<float>();
		vector<Blob<float>*> blob_bottom_vec_;
		vector<Blob<float>*> blob_top_vec_;
		blob_top_vec_.push_back(blob_top_data_);
		blob_top_vec_.push_back(blob_top_label_);

		sum_blob.set_num(1);
		sum_blob.set_channels(FLAGS_channels);
		sum_blob.set_height(FLAGS_height);
		sum_blob.set_width(FLAGS_width);
		const int data_size = FLAGS_channels * FLAGS_height * FLAGS_width;
		for (int i = 0; i < data_size; ++i) {
			sum_blob.add_data(0.);
		}
		LOG(INFO) << "Starting Iteration";

		LayerParameter param;
		ImageDataParameter* image_data_param = param.mutable_image_data_param();
		image_data_param->set_batch_size(FLAGS_batch_size);
		image_data_param->set_new_height(FLAGS_height);
		image_data_param->set_new_width(FLAGS_width);
		if (FLAGS_channels == 1)
			image_data_param->set_is_color(false);
		image_data_param->set_source(argv[1]);
		image_data_param->set_shuffle(false);
		ImageDataLayer<float> layer(param);
		layer.SetUp(blob_bottom_vec_, blob_top_vec_);

		int num_runs = ceil((float)FLAGS_num / (float)FLAGS_batch_size);
		for (int i = 0; i < num_runs; ++i) {
			//One forward pass over batch
			layer.Forward(blob_bottom_vec_, blob_top_vec_);
			
			//We need to check to see if the current batch size is greater than the number of images remaining
			int run_size = FLAGS_batch_size;
			if ((i + 1) * FLAGS_batch_size >= FLAGS_num)
				run_size = FLAGS_num - i * FLAGS_batch_size;
			// Look at each individual example in the batch
			for (int b = 0; b < run_size; ++b) {
				//getMatFromBlobProto(sum_blob, &sum_img);
				int iter = 0;
				for (int chan = 0; chan < FLAGS_channels; chan++) {
					for (int r = 0; r < FLAGS_height; ++r) {
						for (int c = 0; c < FLAGS_width; ++c, iter++) {
							sum_blob.set_data(iter, sum_blob.data(iter) +
								static_cast<float>(*(blob_top_data_->cpu_data() + blob_top_data_->offset(b, chan, r, c))));
						}
					}
				}
				//getMatFromBlobProto(sum_blob, &sum_img2);
				++count;
			}
		}
	}

	for (int i = 0; i < sum_blob.data_size(); ++i) {
		sum_blob.set_data(i, sum_blob.data(i) / count);
	}
	//getMatFromBlobProto(sum_blob, &sum_img_final);
	// Write to disk
	if (argc == 3) {
		LOG(INFO) << "Write to " << argv[2];
		WriteProtoToBinaryFile(sum_blob, argv[2]);
	}
	const int channels = sum_blob.channels();
	const int dim = sum_blob.height() * sum_blob.width();
	std::vector<float> mean_values(channels, 0.0);
	LOG(INFO) << "Number of channels: " << channels;
	for (int c = 0; c < channels; ++c) {
		for (int i = 0; i < dim; ++i) {
			mean_values[c] += sum_blob.data(dim * c + i);
		}
		LOG(INFO) << "mean_value channel [" << c << "]:" << mean_values[c] / dim;
	}
#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
