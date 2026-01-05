CXX = g++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2
EIGEN_PATH ?= /usr/include/eigen3
INCLUDES = -I. -I$(EIGEN_PATH)

BUILD_DIR = build
TEST_DIR = tests

# Category-specific test files
MATH_TESTS = $(BUILD_DIR)/test_01_add $(BUILD_DIR)/test_01_div $(BUILD_DIR)/test_01_mul \
             $(BUILD_DIR)/test_01_neg $(BUILD_DIR)/test_01_pow $(BUILD_DIR)/test_01_sub \
             $(BUILD_DIR)/test_01_exp $(BUILD_DIR)/test_01_log $(BUILD_DIR)/test_01_sqrt \
             $(BUILD_DIR)/test_01_clip

TENSOR_TESTS = $(BUILD_DIR)/test_02_reshape $(BUILD_DIR)/test_02_transpose $(BUILD_DIR)/test_02_flatten \
               $(BUILD_DIR)/test_02_squeeze $(BUILD_DIR)/test_02_unsqueeze $(BUILD_DIR)/test_02_resize \
               $(BUILD_DIR)/test_02_concat $(BUILD_DIR)/test_02_split $(BUILD_DIR)/test_02_slice \
               $(BUILD_DIR)/test_02_gather $(BUILD_DIR)/test_02_scatternd

NN_TESTS = $(BUILD_DIR)/test_03_conv $(BUILD_DIR)/test_03_convtranspose $(BUILD_DIR)/test_03_maxpool \
           $(BUILD_DIR)/test_03_averagepool $(BUILD_DIR)/test_03_globalaveragepool \
           $(BUILD_DIR)/test_03_layernormalization $(BUILD_DIR)/test_03_lstm $(BUILD_DIR)/test_03_gru

ACTIVATION_TESTS = $(BUILD_DIR)/test_04_relu $(BUILD_DIR)/test_04_leakyrelu $(BUILD_DIR)/test_04_elu \
                   $(BUILD_DIR)/test_04_prelu $(BUILD_DIR)/test_04_swish $(BUILD_DIR)/test_04_softmax \
                   $(BUILD_DIR)/test_04_sigmoid $(BUILD_DIR)/test_04_hardsigmoid \
                   $(BUILD_DIR)/test_04_hardswish $(BUILD_DIR)/test_04_tanh

LINALG_TESTS = $(BUILD_DIR)/test_05_matmul $(BUILD_DIR)/test_05_gemm

COMPARE_TESTS = $(BUILD_DIR)/test_06_equal $(BUILD_DIR)/test_06_greater \
                $(BUILD_DIR)/test_06_greaterorequal $(BUILD_DIR)/test_06_less \
                $(BUILD_DIR)/test_06_lessorequal

REDUCE_TESTS = $(BUILD_DIR)/test_07_reducesum $(BUILD_DIR)/test_07_reducemean \
               $(BUILD_DIR)/test_07_reducemax $(BUILD_DIR)/test_07_reducemin \
               $(BUILD_DIR)/test_07_reduceprod $(BUILD_DIR)/test_07_reducel2 \
               $(BUILD_DIR)/test_07_reducel1 $(BUILD_DIR)/test_07_reducesumsquare \
               $(BUILD_DIR)/test_07_reducelogsumexp $(BUILD_DIR)/test_07_reducelogsum

UTILITY_TESTS = $(BUILD_DIR)/test_08_pad

IMAGE_TESTS = $(BUILD_DIR)/test_09_spacetodepth $(BUILD_DIR)/test_09_depthtospace

CONTROL_TESTS = $(BUILD_DIR)/test_10_reversesequence

# All tests
ALL_TESTS = $(MATH_TESTS) $(TENSOR_TESTS) $(NN_TESTS) $(ACTIVATION_TESTS) $(LINALG_TESTS) \
            $(COMPARE_TESTS) $(REDUCE_TESTS) $(UTILITY_TESTS) $(IMAGE_TESTS) $(CONTROL_TESTS)

.PHONY: all clean test test-math test-tensor test-nn test-activation test-linalg \
        test-compare test-reduce test-utility test-image test-control \
        math tensor nn activation linalg compare reduce utility image control

all: $(ALL_TESTS)

# Category targets
math: $(MATH_TESTS)
tensor: $(TENSOR_TESTS)
nn: $(NN_TESTS)
activation: $(ACTIVATION_TESTS)
linalg: $(LINALG_TESTS)
compare: $(COMPARE_TESTS)
reduce: $(REDUCE_TESTS)
utility: $(UTILITY_TESTS)
image: $(IMAGE_TESTS)
control: $(CONTROL_TESTS)

# Build rules
$(BUILD_DIR)/test_%: $(TEST_DIR)/test_%.cpp %.hpp | $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(INCLUDES) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Test targets
test: $(ALL_TESTS)
	@echo "Running all tests..."
	@for test in $(ALL_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "All tests passed!"

test-math: $(MATH_TESTS)
	@echo "Running math tests..."
	@for test in $(MATH_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Math tests passed!"

test-tensor: $(TENSOR_TESTS)
	@echo "Running tensor tests..."
	@for test in $(TENSOR_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Tensor tests passed!"

test-nn: $(NN_TESTS)
	@echo "Running NN tests..."
	@for test in $(NN_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "NN tests passed!"

test-activation: $(ACTIVATION_TESTS)
	@echo "Running activation tests..."
	@for test in $(ACTIVATION_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Activation tests passed!"

test-linalg: $(LINALG_TESTS)
	@echo "Running linear algebra tests..."
	@for test in $(LINALG_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Linear algebra tests passed!"

test-compare: $(COMPARE_TESTS)
	@echo "Running comparison tests..."
	@for test in $(COMPARE_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Comparison tests passed!"

test-reduce: $(REDUCE_TESTS)
	@echo "Running reduction tests..."
	@for test in $(REDUCE_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Reduction tests passed!"

test-utility: $(UTILITY_TESTS)
	@echo "Running utility tests..."
	@for test in $(UTILITY_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Utility tests passed!"

test-image: $(IMAGE_TESTS)
	@echo "Running image tests..."
	@for test in $(IMAGE_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Image tests passed!"

test-control: $(CONTROL_TESTS)
	@echo "Running control flow tests..."
	@for test in $(CONTROL_TESTS); do \
		echo "Running $$test..."; \
		./$$test || exit 1; \
	done
	@echo "Control flow tests passed!"

clean:
	rm -rf $(BUILD_DIR)
