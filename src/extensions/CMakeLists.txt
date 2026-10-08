cmake_minimum_required(VERSION 3.27)

project(_ext LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CUDA_ARCHITECTURES 120)

# Torch's own CMake config rejects this wheel toolkit: nvcc reports 13.2 while
# cuda.h in the same wheels says 13.0. Take the include and library paths directly.
if(NOT Python_EXECUTABLE)
  find_package(Python 3.10 COMPONENTS Interpreter REQUIRED)
endif()
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c
          "from torch.utils.cpp_extension import include_paths, library_paths; print('|'.join(include_paths())); print('|'.join(library_paths()))"
  OUTPUT_STRIP_TRAILING_WHITESPACE
  OUTPUT_VARIABLE TORCH_PATHS
  COMMAND_ERROR_IS_FATAL ANY)
string(REPLACE "\n" ";" TORCH_PATHS "${TORCH_PATHS}")
list(GET TORCH_PATHS 0 TORCH_INCLUDES_JOINED)
list(GET TORCH_PATHS 1 TORCH_LIBRARIES_JOINED)
string(REPLACE "|" ";" TORCH_INCLUDE_DIRS "${TORCH_INCLUDES_JOINED}")
string(REPLACE "|" ";" TORCH_LIBRARY_DIRS "${TORCH_LIBRARIES_JOINED}")

set(OPS rms_norm swiglu rope decode_attention flash_prefill paged_attention kv_quantize)

add_library(_ext SHARED ${CMAKE_CURRENT_LIST_DIR}/bindings.cpp ${CMAKE_CURRENT_LIST_DIR}/src/utils.cpp)
foreach(op ${OPS})
  target_sources(_ext PRIVATE ${CMAKE_CURRENT_LIST_DIR}/src/${op}.cpp ${CMAKE_CURRENT_LIST_DIR}/src/${op}.cu)
endforeach()

target_include_directories(_ext PRIVATE
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${TORCH_INCLUDE_DIRS}
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_link_directories(_ext PRIVATE ${TORCH_LIBRARY_DIRS} /usr/lib/wsl/lib)
target_link_libraries(_ext PRIVATE c10 torch torch_cpu torch_cuda c10_cuda)
target_compile_definitions(_ext PRIVATE _GLIBCXX_USE_CXX11_ABI=1 CCCL_DISABLE_CTK_COMPATIBILITY_CHECK)
target_compile_options(_ext PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)

set_target_properties(_ext PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/mini_vllm_ext
  PREFIX ""
  OUTPUT_NAME "_ext"
  BUILD_RPATH "$ORIGIN;${TORCH_LIBRARY_DIRS}"
  INSTALL_RPATH "$ORIGIN;${TORCH_LIBRARY_DIRS}")
