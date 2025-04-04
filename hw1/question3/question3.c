#include <stdio.h>
#include <math.h>
#include <omp.h>
#include <stdlib.h>

/*
In order to calculate the area under a given curve, we calculate the integral of 
the given function. Integral can also be represented as the summation of very narrow rectangles under the 
curve as it is shown in the figure below.

The algorithm for the calculation is as following:  
• x axis is dived into N portions such that x0, x1, x2 ... xn. 
• The area is of the portion is calculated by using the function and the midpoint  
(i.e. yi = |f((xi+xi+1)/2)|  is calculated).   
• All calculated values are summed (i.e. Result = y0+y1+y2...yn-1).  
In this homework, you are asked to calculate the following integral by dividing the area under the curve 
into 1000000 rectangles and summing the areas of rectangles.  

f(x) = 3e^(x^3) + 9x^2 - 7.14x from -20 to 20.

gcc question3.c -o question3.o -fopenmp -lm
*/

// Define the function f(x) = 3e^(x^3) + 9x^2 - 7.14x
double function(double x) {
    return 3 * exp(pow(x, 3)) + 9 * pow(x, 2) - 7.14 * x;
}

// Midpoint Rule Integration using OpenMP
double midpoint_integral(double a, double b, int n) {
    double width = (b - a) / n; // Width of each rectangle
    double sum = 0.0;
    
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        double mid = a + (i + 0.5) * width; // Midpoint of the subinterval
        sum += function(mid) * width; // Area of rectangle
    }
    
    return sum;
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <num_threads>\n", argv[0]);
        return EXIT_FAILURE;
    }

    double a = -20.0;
    double b = 20.0; 
    int n = 1000000; 

    // Set the number of threads for OpenMP
    omp_set_num_threads(atoi(argv[1]));

    // Calculate the integral
    double start_time = omp_get_wtime();
    double result = midpoint_integral(a, b, n);
    double end_time = omp_get_wtime();

    // Print the result and the time taken
    printf("Approximate integral value: %lf\n", result);
    printf("Time taken for integration: %lf seconds\n", end_time - start_time);

    return 0;
}
