/*
Vector-Matrix Multiplication: Multiplication of two matrices, A and B, produces the matrix C, whose elements ci,j can be computed as follows:
c_(i,j)= ∑_(k=0)^(l-1)▒〖a_(i,k)  × b_(k,j) 〗  
Where A is an (n x l) matrix while B is an (l x m) matrix. 
A vector is a matrix with one column; that is an (n x 1) matrix.  
In this question, your application reads a floating-point matrix and a vector from an input file and stores them to a 2D array and a 1D array. 
Makes the multiplication in parallel and finally stores the result matrix back to an output file. 
Note that the timing will be done only for the multiplication part. 

gcc question2.c -o question2.o -fopenmp
./question2.o a.txt b.txt output.txt generate 1000 1000
./question2.o a.txt b.txt output.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <string.h>
#include <time.h>

void generate_input_files(const char *matrix_file, const char *vector_file, int rows, int cols) {
    printf("Generating matrix file: %s\n", matrix_file);
    printf("Generating vector file: %s\n", vector_file);
    printf("Rows: %d, Cols: %d\n", rows, cols);

    // Seed the random number generator
    srand(time(NULL));

    // Generate matrix A
    FILE *matrix_fp = fopen(matrix_file, "w");
    if (!matrix_fp) {
        perror("Error opening matrix file for writing");
        fprintf(stderr, "Matrix file: %s\n", matrix_file);
        exit(EXIT_FAILURE);
    }

    // Write dimensions of the matrix (rows x cols)
    fprintf(matrix_fp, "%d %d\n", rows, cols); 

    // Fill the matrix with random floats between 0 and 10
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            fprintf(matrix_fp, "%.2f ", (double)rand() / RAND_MAX * 10);
        }
        fprintf(matrix_fp, "\n");
    }
    fclose(matrix_fp);

    // Generate vector B
    FILE *vector_fp = fopen(vector_file, "w");
    if (!vector_fp) {
        perror("Error opening vector file for writing");
        fprintf(stderr, "Vector file: %s\n", vector_file);
        exit(EXIT_FAILURE);
    }

    // Write dimensions of the vector (cols x 1)
    fprintf(vector_fp, "%d 1\n", cols); 

    for (int i = 0; i < cols; i++) {
        fprintf(vector_fp, "%.2f\n", (double)rand() / RAND_MAX * 10); // Random float between 0 and 10
    }

    fclose(vector_fp);
}

void read_file(const char *filename, double ***matrix, int *rows, int *cols) {
    // Open the file for reading
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    // Read the dimensions of the input
    fscanf(file, "%d %d", rows, cols);

    // Allocate memory for a 2D matrix
    *matrix = (double **)malloc((*rows) * sizeof(double *));
    for (int i = 0; i < *rows; i++) {
        (*matrix)[i] = (double *)malloc((*cols) * sizeof(double));
    }

    // Read the matrix data
    for (int i = 0; i < *rows; i++) {
        for (int j = 0; j < *cols; j++) {
            fscanf(file, "%lf", &(*matrix)[i][j]);
        }
    }

    fclose(file);
}

void multiply_matrix_vector(double **matrix, double **vector, double **result, int rows, int cols) {
    // Allocate memory for the result (n x 1 matrix)
    *result = (double *)malloc(rows * sizeof(double));

    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        double sum = 0.0;
        #pragma omp simd reduction(+:sum)
        for (int j = 0; j < cols; j++) {
            sum += matrix[i][j] * vector[j][0];
        }
        (*result)[i] = sum;
    }
}

int main(int argc, char *argv[]) {
    // Argument check
    if (argc < 3 || argc > 6) {
        fprintf(stderr, "Usage: %s <matrix_file> <vector_file> <output_file> <num_threads> or %s generate <rows> <cols> <matrix_file> <vector_file>\n", argv[0], argv[0]);        return EXIT_FAILURE;
    }

    // Generation mode
    if (argc == 6 && strcmp(argv[1], "generate") == 0) {
        int rows = atoi(argv[2]);
        int cols = atoi(argv[3]);

        if (rows <= 0 || cols <= 0) {
            fprintf(stderr, "Invalid dimensions for matrix/vector generation.\n");
            return EXIT_FAILURE;
        }

        generate_input_files(argv[4], argv[5], rows, cols);
        printf("Random matrix and vector files generated successfully.\n");
        return EXIT_SUCCESS;
    }

    // Set the number of threads
    omp_set_num_threads(atoi(argv[4]));


    double **matrix, **vector, *result;
    int rows, cols, vector_rows, vector_cols;

    // Read the matrix
    read_file(argv[1], &matrix, &rows, &cols);

    // Read the vector
    read_file(argv[2], &vector, &vector_rows, &vector_cols);

    // Ensure the dimensions are compatible for multiplication
    if (cols != vector_rows) {
        fprintf(stderr, "Matrix and vector dimensions are incompatible for multiplication.\n");
        return EXIT_FAILURE;
    }

    // Apply the multiplication
    double start_time = omp_get_wtime();
    multiply_matrix_vector(matrix, vector, &result, rows, cols);
    double end_time = omp_get_wtime();

    printf("Multiplication Time: %lf seconds\n", end_time - start_time);

    // Write the result to the output file
    FILE *output_file = fopen(argv[3], "w");
    if (!output_file) {
        perror("Error opening output file");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < rows; i++) {
        fprintf(output_file, "%lf\n", result[i]);
    }

    fclose(output_file);

    // Free allocated memory
    for (int i = 0; i < rows; i++) {
        free(matrix[i]);
    }
    free(matrix);

    for (int i = 0; i < vector_rows; i++) {
        free(vector[i]);
    }

    free(vector);

    free(result);

    return EXIT_SUCCESS;
}