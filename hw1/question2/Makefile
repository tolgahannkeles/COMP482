CC = gcc
CFLAGS = -fopenmp -O3
TARGET = question2.o
SOURCE = question2.c
THREAD_COUNTS = 1 2 4 8 16 32 64
RUNS_PER_THREAD = 5
RESULTS_FILE = benchmark_results.csv
PLOT_SCRIPT = plot_results.py
ROWS = 4000
COLUMNS = 4000
MATRIX_FILE = a.txt
VECTOR_FILE = b.txt
OUTPUT_FILE = output.txt

.PHONY: all clean benchmark plot generate

all: $(TARGET) generate benchmark plot

$(TARGET): $(SOURCE)
	$(CC) $(SOURCE) -o $(TARGET) $(CFLAGS)

generate: $(TARGET)
	./$(TARGET) generate $(ROWS) $(COLUMNS) $(MATRIX_FILE) $(VECTOR_FILE)

benchmark: $(TARGET)
	@echo "Thread count,Run,Time (seconds)" > $(RESULTS_FILE)
	@for thread in $(THREAD_COUNTS); do \
    	for run in $$(seq 1 $(RUNS_PER_THREAD)); do \
            echo "Running with $$thread threads (run $$run of $(RUNS_PER_THREAD))..."; \
            output=$$(./$(TARGET) $(MATRIX_FILE) $(VECTOR_FILE) $(OUTPUT_FILE) $$thread); \
            time=$$(echo "$$output" | grep "Multiplication Time" | awk '{print $$3}'); \
            echo "$$thread,$$run,$$time" >> $(RESULTS_FILE); \
        done; \
    done
	@echo "Benchmark results saved to $(RESULTS_FILE)"

plot: $(RESULTS_FILE)
	@python3 $(PLOT_SCRIPT)

clean:
	rm -f $(TARGET) $(RESULTS_FILE) benchmark_plot.png $(OUTPUT_FILE) $(MATRIX_FILE) $(VECTOR_FILE)