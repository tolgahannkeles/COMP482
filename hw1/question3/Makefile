CC = gcc
CFLAGS = -fopenmp -lm -O3
TARGET = question3.o
SOURCE = question3.c
THREAD_COUNTS = 1 2 4 8 16 32 64
RUNS_PER_THREAD = 5
RESULTS_FILE = benchmark_results.csv
PLOT_SCRIPT = plot_results.py

.PHONY: all clean benchmark plot

all: $(TARGET) benchmark plot

$(TARGET): $(SOURCE)
	$(CC) $(SOURCE) -o $(TARGET) $(CFLAGS)

benchmark: $(TARGET)
	@echo "Thread count,Run,Time (seconds)" > $(RESULTS_FILE)
	@for thread in $(THREAD_COUNTS); do \
        for run in $$(seq 1 $(RUNS_PER_THREAD)); do \
            echo "Running with $$thread threads (run $$run of $(RUNS_PER_THREAD))..."; \
            output=$$(./$(TARGET) $$thread); \
            time=$$(echo "$$output" | grep "Time taken" | awk '{print $$5}'); \
            echo "$$thread,$$run,$$time" >> $(RESULTS_FILE); \
        done; \
    done
	@echo "Benchmark results saved to $(RESULTS_FILE)"

plot: $(RESULTS_FILE)
	@python3 $(PLOT_SCRIPT)

clean:
	rm -f $(TARGET) $(RESULTS_FILE) benchmark_plot.png