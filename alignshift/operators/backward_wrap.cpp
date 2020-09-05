#include <torch/extension.h>

#include <vector>
#include <ATen/ATen.h>
#include <ATen/Config.h>
#include <ATen/NativeFunctions.h>

// at::Tensor cudnn_convolution_backward_input(
//     IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
//     IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
//     bool benchmark, bool deterministic) {
//   AT_ERROR("cudnn_convolution_backward_input: ATen not compiled with cuDNN support");
// }

// at::Tensor cudnn_convolution_backward_weight(
//     IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
//     IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
//     bool benchmark, bool deterministic) {
//   AT_ERROR("cudnn_convolution_backward_weight: ATen not compiled with cuDNN support");
// }

// at::Tensor cudnn_convolution_backward_bias(
//     const at::Tensor& grad_output) {
//   AT_ERROR("cudnn_convolution_backward_bias: ATen not compiled with cuDNN support");
// }

// std::tuple<at::Tensor,at::Tensor,at::Tensor> cudnn_convolution_backward(
//     const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
//     IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
//     bool benchmark, bool deterministic, std::array<bool,3> output_mask) {
//   AT_ERROR("cudnn_convolution_backward: ATen not compiled with cuDNN support");
// }

at::Tensor conv_backward_input(
    at::IntArrayRef input_size,
    const at::Tensor& weight,
    const at::Tensor& grad_output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

    return at::cudnn_convolution_backward_input(
        input_size, grad_output, weight,
        padding, stride, dilation, groups, benchmark, deterministic
    );
}

at::Tensor conv_backward_weight(
    const at::Tensor& input,
    at::IntArrayRef weight_size,
    const at::Tensor& grad_output,
    at::IntArrayRef stride,
    at::IntArrayRef padding,
    at::IntArrayRef dilation,
    int64_t groups,
    bool benchmark,
    bool deterministic) {

    return at::cudnn_convolution_backward_weight(
        weight_size, grad_output, input,
        padding, stride, dilation, groups, benchmark, deterministic
    );
}

at::Tensor conv_backward_bias(
    const at::Tensor& grad_output) {
    
    return at::miopen_convolution_backward_bias(
        grad_output
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("backward_input", &conv_backward_input, "Cudnn convolution input backward");
    m.def("backward_weight", &conv_backward_weight, "Cudnn convolution weight backward");
    m.def("backward_bias", &conv_backward_bias, "Cudnn convolution bias backward");
}
