/**
 * @file    cached_mesh_builder.cpp
 *
 * @author  FULL NAME <xlogin00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using pre-computed field
 *
 * @date    DATE
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "cached_mesh_builder.h"


CachedMeshBuilder::CachedMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Cached")
{

}

unsigned CachedMeshBuilder::marchCubes(const ParametricScalarField &field)
{
  // cached builder based on loop builder
  int totalTriangles = 0;
  // compute total number of cubes in the grid with + 1 in each dimension
  size_t total_cached_cubes_count = (mGridSize + 1) * (mGridSize + 1) * (mGridSize + 1);
  // compute total number of cubes (based on loop solution)
  size_t total_cubes_count = mGridSize * mGridSize * mGridSize;

  // for storing pre-computed sqrts in 1d cache array
  evaluated_values = new float [total_cached_cubes_count];

  const Vec3_t<float> *pPoints = field.getPoints().data();
  const unsigned count = unsigned(field.getPoints().size());

  // loop for pre-computation of coordinates and sqrts
  #pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < total_cached_cubes_count; ++i)
  {
    // compute p coordinates based on 6.4 equation
    float p_value_x = (i % (mGridSize + 1)) * mGridResolution;
    float p_value_y = ((i / (mGridSize + 1)) % (mGridSize + 1)) * mGridResolution;
    float p_value_z = (i / ((mGridSize + 1) * (mGridSize + 1))) * mGridResolution;

    // pre-compute evaluate function

    float value = std::numeric_limits<float>::max();

    for (unsigned j = 0; j < count; ++j)
    {
        float distanceSquared  = (p_value_x - pPoints[j].x) * (p_value_x - pPoints[j].x);
        distanceSquared       += (p_value_y - pPoints[j].y) * (p_value_y - pPoints[j].y);
        distanceSquared       += (p_value_z - pPoints[j].z) * (p_value_z - pPoints[j].z);

        // Comparing squares instead of real distance to avoid unnecessary
        // "sqrt"s in the loop.
        value = std::min(value, distanceSquared);
    }

    float computed_value = sqrt(value);

    // store computed value of sqrt for this coordinates in global 1d array which works as cache
    evaluated_values[i] = computed_value;

  }

  // finally build emited triangles
  #pragma omp parallel for schedule(guided)
  for (size_t i = 0; i < total_cubes_count; ++i)
  {
    // compute 3D position in the grid
    Vec3_t<float> cubeOffset( i % (mGridSize + 1),
                                ( i / (mGridSize + 1)) % (mGridSize + 1),
                                  i / ((mGridSize + 1) * (mGridSize + 1)));

    // evaluate "Marching Cube" at given position in the grid and store the number of triangles generated
    unsigned emitedTriangles = buildCube(cubeOffset, field);

    // count total number of generated triangles 
    #pragma omp critical
    totalTriangles += emitedTriangles;
  }
  return totalTriangles;
}

float CachedMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
  // transform back p coordinates to c coordinates based on 6.5 equation
  float c_back_transformation_x = floor(pos.x / mGridResolution + 0.5);
  float c_back_transformation_y = floor(pos.y / mGridResolution + 0.5) * (mGridSize + 1);
  float c_back_transformation_z = floor(pos.z / mGridResolution + 0.5) * (mGridSize + 1) * (mGridSize + 1);

  // compute access index to 1d cache array of pre-computed values
  int access_index = c_back_transformation_x + c_back_transformation_y + c_back_transformation_z;
  float cached_value = evaluated_values[access_index];
  return cached_value; // get cached value from array of pre-computed values
}

void CachedMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
  // NOTE: This method is called from "buildCube(...)"!

  // Store generated triangle into vector (array) of generated triangles.
  // The pointer to data in this array is return by "getTrianglesArray(...)" call
  // after "marchCubes(...)" call ends.
  #pragma omp critical
  mTriangles.push_back(triangle);

}
