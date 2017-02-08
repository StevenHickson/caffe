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

#ifdef USE_OPENCV
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#endif
using namespace caffe;  // NOLINT(build/namespaces)

using std::min;
using std::pair;
using boost::scoped_ptr;

DEFINE_string(backend, "lmdb",
	"The backend {leveldb, lmdb} containing the images");
DEFINE_bool(use_db, true, "Dictates whether to use a db {leveldb, lmdb} or a list of images and labels");
DEFINE_int32(height, 227, "The new height of images if resizing is needed");
DEFINE_int32(width, 227, "The new width of images if resizing is needed");
DEFINE_int32(channels, 3, "The number of channels");
DEFINE_int32(batch_size, 32, "The batch size");
DEFINE_int32(num, 1, "the total number of images");

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

#ifdef USE_OPENCV
#ifndef GFLAGS_GFLAGS_H_
	namespace gflags = google;
#endif

	gflags::SetUsageMessage("Compute the channel-wise mean and principal "
		" components of a set of images given by a leveldb/lmdb\n"
		"Usage:\n"
		"    compute_image_pca [FLAGS] INPUT_DB [OUTPUT_FILE]\n");

	gflags::ParseCommandLineFlags(&argc, &argv, true);

	/* Let's temporarily remove this since we need to rewrite it
	if (argc < 2 || argc > 3) {
	  gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/compute_image_pca");
	  return 1;
	}*/

	cv::Mat covar;
	cv::Mat covar_prev;
	cv::Mat mean;
	cv::Mat mean_prev;

	int channels;
	int count = 0;

	if (FLAGS_use_db) {
		scoped_ptr<db::DB> db(db::GetDB(FLAGS_backend));
		db->Open(argv[1], db::READ);
		scoped_ptr<db::Cursor> cursor(db->NewCursor());

		Datum first_datum;
		first_datum.ParseFromString(cursor->value());

		if (DecodeDatumNative(&first_datum)) {
			LOG(INFO) << "Decoding Datum";
		}

		const int data_size = first_datum.channels() * first_datum.height()
			* first_datum.width();

		int size_in_datum = std::max<int>(first_datum.data().size(),
			first_datum.float_data_size());
		channels = first_datum.channels();
		const int dim = first_datum.height() * first_datum.width();

		covar = cv::Mat::zeros(channels, channels, CV_64F);
		covar_prev = cv::Mat::zeros(channels, channels, CV_64F);
		mean = cv::Mat::zeros(1, channels, CV_64F);
		mean_prev = cv::Mat::zeros(1, channels, CV_64F);

		// We calculate both mean and covariance online, since for large datasets
		// i.e. Imagenet, we can easily overflow using the naive algorithm.
		// see
		// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
		LOG(INFO) << "Calculating mean and principal components...";
		while (cursor->valid()) {
			Datum datum;
			datum.ParseFromString(cursor->value());
			DecodeDatumNative(&datum);

			size_in_datum = std::max<int>(datum.data().size(),
				datum.float_data_size());
			CHECK_EQ(size_in_datum, data_size) << "Incorrect data field size " <<
				size_in_datum;

			const std::string& data = datum.data();
			if (data.size() != 0) {
				uint64_t n_pixels = 0;
				CHECK_EQ(data.size(), size_in_datum);
				for (int i = 0; i < dim; ++i) {
					n_pixels = count*dim + (i + 1);

					// for each pixel, update mean and covariance for all channels
					for (int c = 0; c < channels; ++c) {
						mean.ptr<double>(0)[c] +=
							(((uint8_t)data[dim * c + i])
								- mean.ptr<double>(0)[c]) / n_pixels;
					}
					for (int c_i = 0; c_i < channels; c_i++) {
						for (int c_j = 0; c_j <= c_i; c_j++) {
							covar.ptr<double>(c_i)[c_j] =
								(covar_prev.ptr<double>(c_i)[c_j] * (n_pixels - 1)
									+ (((uint8_t)data[dim * c_i + i])
										- mean.ptr<double>(0)[c_i])
									* (((uint8_t)data[dim * c_j + i])
										- mean_prev.ptr<double>(0)[c_j])) / n_pixels;
						}
					}

					covar.copyTo(covar_prev);
					mean.copyTo(mean_prev);
				}
			}
			else {
				uint64_t n_pixels = 0;
				CHECK_EQ(datum.float_data_size(), size_in_datum);
				for (int i = 0; i < dim; ++i) {
					n_pixels = count*dim + (i + 1);

					// for each pixel, update mean and covariance for all channels
					for (int c = 0; c < channels; ++c) {
						mean.ptr<double>(0)[c] +=
							(static_cast<float>(datum.float_data(dim * c + i))
								- mean.ptr<double>(0)[c]) / n_pixels;
					}
					for (int c_i = 0; c_i < channels; c_i++) {
						for (int c_j = 0; c_j <= c_i; c_j++) {
							covar.ptr<double>(c_i)[c_j] =
								(covar_prev.ptr<double>(c_i)[c_j] * (n_pixels - 1)
									+ (static_cast<float>(datum.float_data(dim * c_i + i))
										- mean.ptr<double>(0)[c_i])
									* (static_cast<float>(datum.float_data(dim * c_j + i))
										- mean_prev.ptr<double>(0)[c_j])) / n_pixels;
						}
					}

					covar.copyTo(covar_prev);
					mean.copyTo(mean_prev);
				}
			}

			++count;
			if (count % 1000 == 0) {
				LOG(INFO) << "Processed " << count << " files.";
				LOG(INFO) << "Mean" << mean;
				LOG(INFO) << "Sample Covariance" << covar;
			}
			cursor->Next();
		}
	}
	else {
		channels = FLAGS_channels;
		const int dim = FLAGS_height * FLAGS_width;
		const int data_size = dim * channels;

		covar = cv::Mat::zeros(channels, channels, CV_64F);
		covar_prev = cv::Mat::zeros(channels, channels, CV_64F);
		mean = cv::Mat::zeros(1, channels, CV_64F);
		mean_prev = cv::Mat::zeros(1, channels, CV_64F);

		Blob<float>* const blob_top_data_ = new Blob<float>();
		Blob<float>* const blob_top_label_ = new Blob<float>();
		vector<Blob<float>*> blob_bottom_vec_;
		vector<Blob<float>*> blob_top_vec_;
		blob_top_vec_.push_back(blob_top_data_);
		blob_top_vec_.push_back(blob_top_label_);

		// We calculate both mean and covariance online, since for large datasets
		// i.e. Imagenet, we can easily overflow using the naive algorithm.
		// see
		// http://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
		LOG(INFO) << "Calculating mean and principal components...";
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
				uint64_t n_pixels = 0;
				for (int r = 0; r < FLAGS_height; ++r) {
					for (int c = 0; c < FLAGS_width; ++c, ++iter) {
						//for (int i = 0; i < dim; ++i) {
						n_pixels = count*dim + (iter + 1);

						// for each pixel, update mean and covariance for all channels
						for (int chan = 0; chan < FLAGS_channels; chan++) {
							mean.ptr<double>(0)[chan] +=
								(((uint8_t)(blob_top_data_->cpu_data() + blob_top_data_->offset(b, chan, r, c)))
									- mean.ptr<double>(0)[chan]) / n_pixels;
						}
						for (int c_i = 0; c_i < channels; c_i++) {
							for (int c_j = 0; c_j <= c_i; c_j++) {
								covar.ptr<double>(c_i)[c_j] =
									(covar_prev.ptr<double>(c_i)[c_j] * (n_pixels - 1)
										+ (((uint8_t)(blob_top_data_->cpu_data() + blob_top_data_->offset(b, c_i, r, c)))
											- mean.ptr<double>(0)[c_i])
										* (((uint8_t)(blob_top_data_->cpu_data() + blob_top_data_->offset(b, c_j, r, c)))
											- mean_prev.ptr<double>(0)[c_j])) / n_pixels;
							}
						}

						covar.copyTo(covar_prev);
						mean.copyTo(mean_prev);
					}
				}

				++count;
				if (count % 1000 == 0) {
					LOG(INFO) << "Processed " << count << " files.";
					LOG(INFO) << "Mean" << mean;
					LOG(INFO) << "Sample Covariance" << covar;
				}
			}
		}
	}

	// fill in uncalculated symmetric part of matrix
	for (int c_i = 0; c_i < channels; c_i++) {
		for (int c_j = 0; c_j < c_i; c_j++) {
			covar.ptr<double>(c_j)[c_i] = covar.ptr<double>(c_i)[c_j];
		}
	}

	LOG(INFO) << "Processed " << count << " files.";
	LOG(INFO) << "Mean channel values: " << mean;
	LOG(INFO) << "Channel Covariance: " << covar;

	cv::Mat eigenvalues, eigenvectors;
	cv::eigen(covar, eigenvalues, eigenvectors);

	for (int c = 0; c < channels; ++c) {
		LOG(INFO) << "mean_value: " << mean.ptr<double>(0)[c];
	}
	for (int c = 0; c < channels; ++c) {
		LOG(INFO) << "eigen_value: " << eigenvalues.ptr<double>(0)[c];
	}
	for (int i = 0; i < channels; ++i) {
		for (int j = 0; j < channels; ++j) {
			LOG(INFO) << "eigen_vector_component: " << eigenvectors.ptr<double>(i)[j];
		}
	}

#else
	LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
	return 0;
}
