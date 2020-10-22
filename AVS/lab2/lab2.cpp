/**
 * @file      lab2.cpp
 *
 * @author    Jiri Jaros \n
 *            Faculty of Information Technology \n
 *            Brno University of Technology \n
 *            jarosjir@fit.vutbr.cz
 *
 * @brief     AVS - PC lab 2
 *            Matrix multiplication
 *
 * @version   2021
 *
 * @date      18 October   2020, 19:20 (created) \n
 * @date      18 October   2020, 19:20 (created) \n
 *
 */

#include <unistd.h>
#include <immintrin.h>
#include <cstdio>
#include <map>
#include <chrono>
#include <functional>
#include <mkl.h>
#include <cmath>

using namespace std;

//--------------------------------------------------------------------------------------------------------------------//
//                                     Type definition Function prototypes                                            //
//--------------------------------------------------------------------------------------------------------------------//

/**
 * @struct MatrixSizes
 *
 * All necessary sizes for the matrix-matrix multiplication
 */
struct MatrixSizes
{
  constexpr MatrixSizes(size_t n, size_t m, size_t p) : N(n), M(m), P(p) {};

  /// Number of rows of the first matrix
  const size_t N;
  /// Number of columns of the first matrix / number of rows of the first matrix
  const size_t M;
  /// Number of columns of the second matrix
  const size_t P;

  friend bool operator<(const MatrixSizes& l, const MatrixSizes& r)
  {
    return (((l.N * l.M) + (l.M * l.P) + (l.N * l.P)) < ((r.N * r.M) + (r.M * r.P) + (r.N * r.P)));
  }
};// end of MatrixSizes
//----------------------------------------------------------------------------------------------------------------------

/**
 * Function prototype of matrix - matrix multiplication
 */
template<size_t N,
         size_t M,
         size_t P>
void matrixMul(float* c, const float* a, const float* b);

/// Function pointer definition
using MatrixBenchmarkFnc = std::function<void(float*, const float*, const float*)>;

//--------------------------------------------------------------------------------------------------------------------//
//                                     Global constants and benchmark maps                                            //
//--------------------------------------------------------------------------------------------------------------------//
// Number of tests
constexpr size_t nTests = 7;
// Size of arrays used
constexpr MatrixSizes benchmarkSizes[nTests] = {
          //  [N * M] x [M * P]
          //     N     M     P
  MatrixSizes(  16,   16,   16),
  MatrixSizes( 128,  128,  128),
  MatrixSizes( 130,  130,  130),
  MatrixSizes( 250,  250,  250),
  MatrixSizes( 256,  256,  256),
  MatrixSizes(1024, 1024, 1024),
  MatrixSizes(1536, 1536, 1536)
};

// Number of benchmark repetitions
constexpr size_t testRept[nTests]   = {10000, 1000, 1000, 300, 300, 2, 1};
// Max array size
const MatrixSizes maxSizes = benchmarkSizes[nTests - 1];

// Map with benchmark sizes
std::map<MatrixSizes, MatrixBenchmarkFnc> matrixBenchmarks =
{
  {benchmarkSizes[0], &matrixMul<benchmarkSizes[0].N, benchmarkSizes[0].M, benchmarkSizes[0].P>},
  {benchmarkSizes[1], &matrixMul<benchmarkSizes[1].N, benchmarkSizes[1].M, benchmarkSizes[1].P>},
  {benchmarkSizes[2], &matrixMul<benchmarkSizes[2].N, benchmarkSizes[2].M, benchmarkSizes[2].P>},
  {benchmarkSizes[3], &matrixMul<benchmarkSizes[3].N, benchmarkSizes[3].M, benchmarkSizes[3].P>},
  {benchmarkSizes[4], &matrixMul<benchmarkSizes[4].N, benchmarkSizes[4].M, benchmarkSizes[4].P>},
  {benchmarkSizes[5], &matrixMul<benchmarkSizes[5].N, benchmarkSizes[5].M, benchmarkSizes[5].P>},
  {benchmarkSizes[6], &matrixMul<benchmarkSizes[6].N, benchmarkSizes[6].M, benchmarkSizes[6].P>},
};// end of matrixBenchmarks

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//                                           Routines to be implemented                                               //
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/**
 * Matrix multiplication c[N * P] = a[N * M] * b[M * P]
 *
 * @tparam N   - Number of rows of the first matrix
 * @tparam M   - Number of columns of the first matrix / number of rows of the first matrix
 * @tparam P   - Number of columns of the second matrix
 *
 * @param [out] c - Output matrix
 * @param [in]  a - Input matrix a
 * @param [in]  b - Input matrix b
 */
template<size_t N,
         size_t M,
         size_t P>
void matrixMul(float*       c,
               const float* a,
               const float* b)
{
  // Go over all rows in the result matrix
  // prohozeni smycek, nulovani zvlast
  // odstraneni datove zavislosti
  for (size_t i = 0; i < N * P; i++)
  {
    c[i] = 0.0f;
  }
  for (size_t i = 0; i < N; i++)
  {
    // Go over all cols in the result matrix
    for (size_t k = 0; k < M; k++)
    {
      // Zero the current cell

      // Calculate vector dot product of a given row and col
      // rozbaleni smycky nekolikrat pres k a iterace nekolika cyklu zaroven

      for (size_t j = 0; j < P; j++)
      {
        c[i * P + j] += a[i * M + k] * b[k * P + j];
      }
    }
  }
}// end of matrixMul
//----------------------------------------------------------------------------------------------------------------------

/**
 * Allocate memory
 * @param  [in] height  - Number of rows
 * @param  [in] width   - Number of cols
 * @return Pointer to allocated memory
 */
float* allocateMemory(size_t height,
                      size_t width)
{
  return ((float *) malloc(height * width * sizeof(float)));
}// end of allocateMemory
//----------------------------------------------------------------------------------------------------------------------

/**
 * Free memory
 * @param [in] matrix - Matrix to be fried.
 */
void freeMemory(float* matrix)
{
  free(matrix);
}// end of freeMemory
//----------------------------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/**
 * Generate random data to fill the arrays
 * @param [out] array - Array to be filled in
 * @param [in] size   - Size of the array
 */
void generateData(float* matrix,
                  size_t height,
                  size_t width)
{
  for (size_t i = 0; i < height * width; i++)
  {
    matrix[i] = ((float)(rand() % 20) - 5) / 5.0f;
  }
}// end of generateData
//----------------------------------------------------------------------------------------------------------------------

/**
 * Find maximum absolute error
 * @param [in] a - First matrix
 * @param [in] b - Second  matrix
 * @param height - Number of rows
 * @param width  - Number of cols
 * @return Maximum absolute error over all cells
 */
float maxError(const float* a,
               const float* b,
               size_t       height,
               size_t       width)
{
  float error = 0.f;

  for (size_t i = 0; i < height * width; i++)
  {
    error = std::max(error, fabs(a[i] - b[i]));
  }
  return error;
}// end of maxError
//----------------------------------------------------------------------------------------------------------------------

/**
 * Find maximum absolute values
 * @param [in] a - Input matrix
 * @param height - Number of rows
 * @param width  - Number of cols
 * @return Maximum absolute value over all cells
 */
float maxValue(float* a,
               size_t height,
               size_t width)
{
  float maxVal = 0.0f;

  for (size_t i = 0; i < width * height; i++)
  {
    maxVal = std::max(maxVal, fabs(a[i]));
  }
  return maxVal;
}// end of maxValue
//----------------------------------------------------------------------------------------------------------------------


/**
 *  main function
 */
int main(int argc, char** argv)
{
  char hostName[31];
  gethostname(hostName, 30);
  printf("---------------------------------------------------\n");
  printf(" Efficient Implementation of Matrix Multiplication\n");
  printf(" Running on: %s\n", hostName);
  printf("---------------------------------------------------\n");

  //------------------------------------------------------------------------------------------------------------------//
  // Memory allocation                                                                                                //
  //------------------------------------------------------------------------------------------------------------------//
  float* a = allocateMemory(maxSizes.N, maxSizes.M);
  float* b = allocateMemory(maxSizes.M, maxSizes.P);
  float* c = allocateMemory(maxSizes.N, maxSizes.P);
  float* d = allocateMemory(maxSizes.N, maxSizes.P);

  if (!(a && b && c && d))
  {
    printf(" Allocation failure\n");
    exit(EXIT_FAILURE);
  }

  // Generate input matrices
  printf("Generating random data... "); fflush(stdout);
  generateData(a, maxSizes.N, maxSizes.M);
  generateData(b, maxSizes.M, maxSizes.P);
  printf("Done\n");

  //------------------------------------------------------------------------------------------------------------------//
  // Test: Matrix multiplication                                                                                      //
  //------------------------------------------------------------------------------------------------------------------//

  printf("Matrix multiplication tests:\n");
  for (size_t testId = 0; testId < nTests; testId++)
  {
    float avsGflops = 0.f;
    float mklGflops = 0.f;

    const size_t nElements = benchmarkSizes[testId].N * benchmarkSizes[testId].M * benchmarkSizes[testId].P;

    /// AVS version
    {
      printf("  -- Testing matrices [%4d, %4d] x [%4d, %4d] --\n",
          benchmarkSizes[testId].N, benchmarkSizes[testId].M, benchmarkSizes[testId].M, benchmarkSizes[testId].P);
      printf ("  - Number of repetitions         = %9.3f\n", float(testRept[testId]));

      auto startTime = std::chrono::high_resolution_clock::now();

      // Run benchmark
      for (size_t rept = 0; rept < testRept[testId]; rept++)
      {
        #pragma noinline recursive
        matrixBenchmarks[benchmarkSizes[testId]](c, a, b);
      }

      // Elapsed time
      const auto   endTime = std::chrono::high_resolution_clock::now();
      const double time    = (endTime - startTime) / std::chrono::milliseconds(1);

      // Flops
      avsGflops = (float(2 * nElements) / time) / (1000000.f / testRept[testId]);

      printf ("  - AVS: Time to calculate result = %9.3f ms\n", time / testRept[testId]);
      printf ("  - AVS: Performance              = %9.3f GFLOPS\n\n", avsGflops);
    }

    /// MKL version
    {
      auto start_time = std::chrono::high_resolution_clock::now();

      // Run benchmark
      for (size_t rept = 0; rept < testRept[testId]; rept++)
      {
        #pragma noinline recursive
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    benchmarkSizes[testId].N, benchmarkSizes[testId].M, benchmarkSizes[testId].P,
                    1.0f, a, benchmarkSizes[testId].M, b, benchmarkSizes[testId].P, 0.0f, d, benchmarkSizes[testId].P);
      }

      // Elapsed time
      const auto   endTime = std::chrono::high_resolution_clock::now();
      const double time    = (endTime - start_time) / std::chrono::milliseconds(1);

      // Flops
      mklGflops = (float(2 * nElements) / time) / (1000000.f / testRept[testId]);

      printf ("  - MKL: Time to calculate result = %9.3f ms\n", time / testRept[testId]);
      printf ("  - MKL: Performance              = %9.3f GFLOPS\n", mklGflops);
    }

    const float reference = maxValue(c, benchmarkSizes[testId].N, benchmarkSizes[testId].P);
    const float absError  = maxError(c, d, benchmarkSizes[testId].N, benchmarkSizes[testId].P);
    const float relError  = absError / reference;

    printf("  --------------------------------------------------\n");
    printf("  - Max absolute error =  %8.3e \n", absError);
    printf("  - Max relative error =  %8.3e \n", relError);
    printf("  - MKL vs AVS perf    =  %8.3fx \n", mklGflops / avsGflops);
    printf("  --------------------------------------------------\n\n");
  }

  // Free memory
  freeMemory(a);
  freeMemory(b);
  freeMemory(c);
  freeMemory(d);

  return EXIT_SUCCESS;
}// end of main
//----------------------------------------------------------------------------------------------------------------------
