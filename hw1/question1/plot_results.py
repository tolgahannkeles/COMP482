import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the benchmark results
df = pd.read_csv('benchmark_results.csv')

# Calculate averages and standard deviations for each thread count
avg_times = df.groupby('Thread count')['Time (seconds)'].mean()
std_times = df.groupby('Thread count')['Time (seconds)'].std()

# Calculate speedup (relative to single-thread performance)
base_time = avg_times[1]  # Time for single thread
speedup = base_time / avg_times

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Execution time vs. Thread count
thread_counts = avg_times.index
ax1.errorbar(thread_counts, avg_times, yerr=std_times, marker='o', linestyle='-', capsize=5)
ax1.set_xscale('log', base=2)  # Log scale for x-axis
ax1.set_xlabel('Number of Threads')
ax1.set_ylabel('Average Execution Time (seconds)')
ax1.set_title('Histogram Generation Performance')
ax1.grid(True)

# Plot 2: Speedup vs. Thread count
ax2.plot(thread_counts, speedup, marker='o', linestyle='-')
ax2.plot(thread_counts, thread_counts, 'k--', alpha=0.7, label='Ideal Speedup')  # Ideal speedup reference line
ax2.set_xscale('log', base=2)  # Log scale for x-axis
ax2.set_xlabel('Number of Threads')
ax2.set_ylabel('Speedup (relative to 1 thread)')
ax2.set_title('Performance Scaling')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.savefig('benchmark_plot.png')
print("Plot saved to benchmark_plot.png")

# Print summary statistics
print("\nPerformance Summary:")
print("====================")
summary = pd.DataFrame({
    'Avg Time (s)': avg_times,
    'Std Dev (s)': std_times,
    'Speedup': speedup,
    'Efficiency (%)': (speedup / thread_counts) * 100
})
print(summary)