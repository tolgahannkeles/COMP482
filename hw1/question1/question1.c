/*
You need to write an application that reads a large quantity of floating-point data 
which are between [0.0-5.0] from an input file and stores those numbers into an array. 
Then, the application makes a histogram of the data by simply dividing the range of the data up into five equal-sized bins. 
Note that taking input and printing output can be done serially. The time measurement will be done only for the histogram generation part. 
Ex: 2.9 1.3 0.4 1.3 0.3 4.4 1.7 3.2 0.4 0.3 4.9 2.4 
Output: 
[0-1)   4
[1-2)   3
[2-3)   2
[3-4)   1
[4-5]   2

gcc question1.c -o question1.o -fopenmp
./question1.o input.txt
*/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <string.h>

#define NUM_BINS 5
#define MAX_VALUE 5.0
#define MAX_NUM 1000000

void generate_input_file(const char *filename, int num_samples) {
    // Open the file to write
    FILE *file = fopen(filename, "w");
    if (!file) {
        perror("Error creating input file");
        return;
    }
    // Generate random floating-point numbers between 0.0 and MAX_VALUE
    srand(time(NULL));
    for (int i = 0; i < num_samples; i++) {
        fprintf(file, "%.2f ", (double)rand() / RAND_MAX * MAX_VALUE);
    }
    fclose(file);
    printf("Input file '%s' generated with %d samples.\n", filename, num_samples);
}

int main(int argc, char *argv[]) {
    // Argument check
    if (argc < 2 || argc > 4) {
        printf("Invalid number of arguments: %d\n", argc);
        printf("Usage: %s generate <input_file> <size> or %s <input_file> <number_of_thread>\n", argv[0], argv[0]);
        printf("\tExample: %s input.txt 4\n", argv[0]);
        printf("\tExample: %s generate input.txt\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Input file generation
    if (argc == 4 && strcmp(argv[1], "generate") == 0) {
        printf("Generating input file...\n");
        generate_input_file(argv[2], atoi(argv[3]));
        return EXIT_SUCCESS;
    }

    // Open the input file
    FILE *file = fopen(argv[1], "r");
    if (!file) {
        perror("Error opening file");
        return EXIT_FAILURE;
    }

    int num_threads = atoi(argv[2]);

    // Allocate memory for histogram
    double num;
    int* histogram = (int*)calloc(NUM_BINS, sizeof(int));
    if (!histogram) {
        perror("Memory allocation failed");
        fclose(file);
        return EXIT_FAILURE;
    }

    // Allocate memory for numbers
    double *numbers = (double *)malloc(sizeof(double) * MAX_NUM);
    if (!numbers) {
        perror("Memory allocation failed");
        fclose(file);
        free(histogram);
        return EXIT_FAILURE;
    }

    // Read numbers from the file
    int count = 0;
    while (fscanf(file, "%lf", &num) == 1) {
        numbers[count++] = num;
    }
    fclose(file);

    // Set the number of threads 
    omp_set_num_threads(num_threads);

    double start_time = omp_get_wtime();

    #pragma omp parallel
    {
        // Each thread will have its own local histogram
        int local_histogram[NUM_BINS] = {0};


        #pragma omp for nowait
        for (int i = 0; i < count; i++) {
            // Calculate the bin index for each number
            // and update the local histogram
            int bin = (int)(numbers[i] / (MAX_VALUE / NUM_BINS));
            if (bin >= NUM_BINS) bin = NUM_BINS - 1;
            local_histogram[bin]++;
        }

        // Merge local histograms into the global histogram
        #pragma omp critical
        for (int i = 0; i < NUM_BINS; i++) {
            histogram[i] += local_histogram[i];
        }
    }
    
    double end_time = omp_get_wtime();
    /*
    0.058276
    #pragma omp parallel for
    for (int i = 0; i < count; i++) {
        int bin = (int)(numbers[i] / (MAX_VALUE / NUM_BINS));
        if (bin >= NUM_BINS) bin = NUM_BINS - 1;
        #pragma omp atomic
        histogram[bin]++;
    }
    */

    // Print histogram
    printf("[0-1)   %d\n", histogram[0]);
    printf("[1-2)   %d\n", histogram[1]);
    printf("[2-3)   %d\n", histogram[2]);
    printf("[3-4)   %d\n", histogram[3]);
    printf("[4-5]   %d\n", histogram[4]);

    printf("Histogram computation time: %f seconds\n", end_time - start_time);

    free(histogram);
    free(numbers);
    return 0;
}
