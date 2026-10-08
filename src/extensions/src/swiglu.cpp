#include "mini_vllm_ext.h"

namespace mini_vllm_ext {

at::Tensor swiglu(const at::Tensor& gate, const at::Tensor& up) {
  check_cuda(gate, "gate");
  check_cuda(up, "up");
  check_floating(gate, "gate");
  check_same_dtype(gate, up, "gate", "up");
  TORCH_CHECK(gate.sizes() == up.sizes(), "gate and up must have the same shape");

  const at::Tensor gate_input = gate.contiguous();
  const at::Tensor up_input = up.contiguous();
  at::Tensor out = at::empty(gate_input.sizes(), gate_input.options());
  if (gate_input.numel() == 0) {
    return out;
  }
  launch_swiglu(gate_input, up_input, out);
  return out;
}

}  // namespace mini_vllm_ext
