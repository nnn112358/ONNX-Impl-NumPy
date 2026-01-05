# ONNX Operators C++ Implementation Makefile
# Eigen-based header-only implementation

# Compiler settings
CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2

# Eigen path - modify this if Eigen is installed in a different location
# Common locations: /usr/include/eigen3, /usr/local/include/eigen3, ./eigen
EIGEN_PATH = /usr/include/eigen3

# Check alternative Eigen locations if not found
ifeq ($(wildcard $(EIGEN_PATH)/Eigen/Core),)
    ifneq ($(wildcard /usr/local/include/eigen3/Eigen/Core),)
        EIGEN_PATH = /usr/local/include/eigen3
    else ifneq ($(wildcard ./eigen/Eigen/Core),)
        EIGEN_PATH = ./eigen
    else
        $(warning Eigen not found. Please install Eigen or set EIGEN_PATH)
    endif
endif

# Include paths
INCLUDES = -I. -I$(EIGEN_PATH)

# Build directory
BUILD_DIR = build

# Source directories
CPP_DIR = cpp
TEST_DIR = $(CPP_DIR)/tests

# Test executables - Math operations (Category 01)
MATH_TESTS = test_01_add test_01_div test_01_mul test_01_neg test_01_pow \
             test_01_sub test_01_exp test_01_log test_01_sqrt test_01_clip

# Test executables - Tensor operations (Category 02)
TENSOR_TESTS = test_02_reshape test_02_transpose test_02_flatten test_02_squeeze \
               test_02_unsqueeze test_02_resize test_02_concat test_02_split \
               test_02_slice test_02_gather test_02_scatternd

# Test executables - Neural network layers (Category 03)
NN_TESTS = test_03_conv test_03_convtranspose test_03_maxpool test_03_averagepool \
           test_03_globalaveragepool test_03_layernormalization test_03_lstm test_03_gru

# Test executables - Activation functions (Category 04)
ACTIVATION_TESTS = test_04_relu test_04_leakyrelu test_04_elu test_04_prelu \
                   test_04_swish test_04_softmax test_04_sigmoid test_04_hardsigmoid \
                   test_04_hardswish test_04_tanh

# Test executables - Linear algebra (Category 05)
LINALG_TESTS = test_05_matmul test_05_gemm

# Test executables - Comparison operations (Category 06)
COMPARE_TESTS = test_06_equal test_06_greater test_06_greaterorequal \
                test_06_less test_06_lessorequal

# Test executables - Reduction operations (Category 07)
REDUCE_TESTS = test_07_reducesum test_07_reducemean test_07_reducemax test_07_reducemin \
               test_07_reduceprod test_07_reducel2 test_07_reducel1 test_07_reducesumsquare \
               test_07_reducelogsumexp test_07_reducelogsum

# Test executables - Utility (Category 08)
UTIL_TESTS = test_08_pad

# Test executables - Image processing (Category 09)
IMAGE_TESTS = test_09_spacetodepth test_09_depthtospace

# Test executables - Control flow (Category 10)
CONTROL_TESTS = test_10_reversesequence

# All test executables
ALL_TESTS = $(MATH_TESTS) $(TENSOR_TESTS) $(NN_TESTS) $(ACTIVATION_TESTS) \
            $(LINALG_TESTS) $(COMPARE_TESTS) $(REDUCE_TESTS) $(UTIL_TESTS) \
            $(IMAGE_TESTS) $(CONTROL_TESTS)

# Add build directory prefix
TEST_BINS = $(addprefix $(BUILD_DIR)/, $(ALL_TESTS))

# Default target
.PHONY: all
all: $(BUILD_DIR) $(TEST_BINS)

# Create build directory
$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Pattern rule for building test executables
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cpp $(CPP_DIR)/%.hpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

# Category-specific targets
.PHONY: math tensor nn activation linalg compare reduce util image control

math: $(addprefix $(BUILD_DIR)/, $(MATH_TESTS))
tensor: $(addprefix $(BUILD_DIR)/, $(TENSOR_TESTS))
nn: $(addprefix $(BUILD_DIR)/, $(NN_TESTS))
activation: $(addprefix $(BUILD_DIR)/, $(ACTIVATION_TESTS))
linalg: $(addprefix $(BUILD_DIR)/, $(LINALG_TESTS))
compare: $(addprefix $(BUILD_DIR)/, $(COMPARE_TESTS))
reduce: $(addprefix $(BUILD_DIR)/, $(REDUCE_TESTS))
util: $(addprefix $(BUILD_DIR)/, $(UTIL_TESTS))
image: $(addprefix $(BUILD_DIR)/, $(IMAGE_TESTS))
control: $(addprefix $(BUILD_DIR)/, $(CONTROL_TESTS))

# Run all tests
.PHONY: test
test: all
	@echo "Running all tests..."
	@for test in $(TEST_BINS); do \
		echo ""; \
		echo "=== Running $$test ==="; \
		$$test || exit 1; \
	done
	@echo ""
	@echo "All tests passed!"

# Run specific category tests
.PHONY: test-math test-tensor test-nn test-activation test-linalg test-compare test-reduce

test-math: math
	@for test in $(addprefix $(BUILD_DIR)/, $(MATH_TESTS)); do \
		echo "=== $$test ==="; $$test; \
	done

test-tensor: tensor
	@for test in $(addprefix $(BUILD_DIR)/, $(TENSOR_TESTS)); do \
		echo "=== $$test ==="; $$test; \
	done

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*

# Help target
.PHONY: help
help:
	@echo "ONNX Operators C++ Implementation - Makefile Help"
	@echo ""
	@echo "Prerequisites:"
	@echo "  - C++17 compatible compiler (g++ 7+, clang++ 5+)"
	@echo "  - Eigen library (header-only)"
	@echo ""
	@echo "Installation:"
	@echo "  Ubuntu/Debian: sudo apt-get install libeigen3-dev"
	@echo "  macOS:         brew install eigen"
	@echo "  Or download from: https://eigen.tuxfamily.org"
	@echo ""
	@echo "Targets:"
	@echo "  all        - Build all test executables (default)"
	@echo "  math       - Build math operation tests"
	@echo "  tensor     - Build tensor operation tests"
	@echo "  nn         - Build neural network layer tests"
	@echo "  activation - Build activation function tests"
	@echo "  linalg     - Build linear algebra tests"
	@echo "  compare    - Build comparison operation tests"
	@echo "  reduce     - Build reduction operation tests"
	@echo "  util       - Build utility tests"
	@echo "  image      - Build image processing tests"
	@echo "  control    - Build control flow tests"
	@echo "  test       - Run all tests"
	@echo "  test-math  - Run math operation tests"
	@echo "  clean      - Remove build artifacts"
	@echo "  help       - Show this help message"
	@echo ""
	@echo "Usage examples:"
	@echo "  make               # Build all tests"
	@echo "  make math          # Build only math tests"
	@echo "  make test          # Build and run all tests"
	@echo "  make test-math     # Build and run math tests"
	@echo "  make clean         # Clean build directory"
	@echo ""
	@echo "Custom Eigen path:"
	@echo "  make EIGEN_PATH=/path/to/eigen"
