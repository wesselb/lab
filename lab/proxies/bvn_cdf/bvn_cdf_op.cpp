#include <iostream>
#include <thread>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace std;

extern "C" {
    double bvnd_(double* x,
                 double* y,
                 double* rho);
}

// Convenient alias for Eigen::Tensor
template <typename T>
using FlatTensor = typename tensorflow::TTypes<T>::Flat;

// Convenient alias for tensorflow::Tensor
typedef tensorflow::Tensor Tensor;

template <typename T>
void bvn_cdf(FlatTensor<T>* prob,
             const int start,
             const int length,
             FlatTensor<const T>* x,
             FlatTensor<const T>* y,
             FlatTensor<const T>* rho) {
    double this_x, this_y, this_rho;
    for (int i = start; i < start + length; i++) {
        this_x = -(*x)(i);
        this_y = -(*y)(i);
        this_rho = (*rho)(i);
        (*prob)(i) = bvnd_(&this_x, &this_y, &this_rho);
    }
}

template <typename T>
void bvn_cdf_parallel(FlatTensor<T>* prob,
                      const int length,
                      FlatTensor<const T>* x,
                      FlatTensor<const T>* y,
                      FlatTensor<const T>* rho) {
    int num_threads = std::thread::hardware_concurrency();
    int segment_length = length / num_threads;
    std::thread* ts = new std::thread[num_threads];

    int cur_pos = 0;
    for (int i = 0; i < num_threads; i++) {
        // Determine length of segment.
        int this_length = segment_length;
        if (i == num_threads - 1) this_length = length - cur_pos;

        // Launch thread.
        ts[i] = std::thread(bvn_cdf<T>,
                            prob,
                            cur_pos,
                            this_length,
                            x,
                            y,
                            rho);

        cur_pos += segment_length;
    }

    // Wait for threads to finish.
    for (int i = 0; i < num_threads; i++) ts[i].join();

    delete [] ts;
}

REGISTER_OP("BvnCdf")
    .Input("x: T")
    .Input("y: T")
    .Input("rho: T")
    .Output("prob: T")
    .Attr("T: {float, double}")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
        // Check ranks of inputs.
        tensorflow::shape_inference::ShapeHandle x_s, y_s, rho_s;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &x_s));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 1, &y_s));
        TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &rho_s));

        // Set shape of output to that of input.
        c->set_output(0, c->input(0));

        return tensorflow::Status::OK();
    });

class BvnCdfOp : public tensorflow::OpKernel {
    public:
        explicit BvnCdfOp(tensorflow::OpKernelConstruction* c)
            : tensorflow::OpKernel(c) {}

    void Compute(tensorflow::OpKernelContext* c) override {
        // Grab the inputs.
        const Tensor& x_t = c->input(0);
        const Tensor& y_t = c->input(1);
        const Tensor& rho_t = c->input(2);

        // Tensors must have equal numbers of elements.
        OP_REQUIRES(c, x_t.dim_size(0) == y_t.dim_size(0)
                       && y_t.dim_size(0) == rho_t.dim_size(0),
                    tensorflow::errors::InvalidArgument(
                        "Inputs must be of equal length."));

        // Create output tensor.
        Tensor* out_t = NULL;
        OP_REQUIRES_OK(c, c->allocate_output(0, x_t.shape(), &out_t));

        // Perform computation, dispatching on data type.
        if (x_t.dtype() == tensorflow::DataType::DT_DOUBLE)
            this->ComputeTyped<double>(c, &x_t, &y_t, &rho_t, out_t);
        else
            this->ComputeTyped<float>(c, &x_t, &y_t, &rho_t, out_t);
    }

    template <typename T>
    void ComputeTyped(tensorflow::OpKernelContext* c,
                      const Tensor* x_t,
                      const Tensor* y_t,
                      const Tensor* rho_t,
                      Tensor* out_t) {
        FlatTensor<const T> x = x_t->flat<T>();
        FlatTensor<const T> y = y_t->flat<T>();
        FlatTensor<const T> rho = rho_t->flat<T>();
        FlatTensor<T> out = out_t->flat<T>();

        // Verify that correlation coefficients are between -1 and 1.
        for (int i = 0; i < rho.size(); i++)
            OP_REQUIRES(c, -1 <= rho(i) && rho(i) <= 1,
                        tensorflow::errors::InvalidArgument(
                            "Correlation coefficients must be between -1 "
                            "and 1."));

        bvn_cdf_parallel<T>(&out, x.size(), &x, &y, &rho);
    }
};

REGISTER_KERNEL_BUILDER(Name("BvnCdf").Device(tensorflow::DEVICE_CPU), BvnCdfOp);