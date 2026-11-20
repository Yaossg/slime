.PHONY: all examples bench test test-cpu test-gpu clean help
all: examples bench
examples:
	$(MAKE) -C examples
bench:
	$(MAKE) -C bench
test: test-cpu
	$(MAKE) -C bench test
test-cpu:
	$(MAKE) -C examples test-cpu
test-gpu:
	$(MAKE) -C examples test-gpu
	$(MAKE) -C bench test-gpu
clean:
	$(MAKE) -C examples clean
	$(MAKE) -C bench clean
help:
	@echo 'make            Build blog examples and the pattern search CLI'
	@echo 'make test-cpu   Test forward examples without CUDA Toolkit or a GPU'
	@echo 'make test       Run all host checks (nvcc required, GPU not required)'
	@echo 'make test-gpu   Verify forward, reverse, reuse and CLI results on a GPU'
	@echo 'make clean      Remove compiled outputs'
