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
  // 1. Compute total number of cubes in the grid.
  int totalCubesCount = (mGridSize+1)*(mGridSize+1)*(mGridSize+1);
  std::vector <float> array_of_coordinates[totalCubesCount];
  unsigned totalTriangles = 0;

  for (int i = 0; i < totalCubesCount; ++i)
  {

    auto p_value = (i) * mGridResolution;
    auto c_back_transformation = (p_value / mGridResolution) + (1/2);

    auto p_value_x = i % (mGridSize+1) * mGridResolution;
    auto p_value_y = (i / (mGridSize+1)) % (mGridSize+1) * mGridResolution;
    auto p_value_z = i / ((mGridSize+1)*(mGridSize +1)) *mGridResolution;

    auto c_back_transformation_x = (p_value_x / mGridResolution) + (1/2);
    auto c_back_transformation_y = (p_value_y / mGridResolution) + (1/2);
    auto c_back_transformation_z = (p_value_z / mGridResolution) + (1/2);





    array_of_coordinates[i].push_back(c_back_transformation_x);
    array_of_coordinates[i].push_back(c_back_transformation_y);
    array_of_coordinates[i].push_back(c_back_transformation_z);

  }

  for (int i = 0; i < totalCubesCount; ++i)
  {
    Vec3_t<float> cubeOffset( array_of_coordinates[i][0],
                             array_of_coordinates[i][1],
                              array_of_coordinates[i][2]);
    unsigned emitedTriangles = buildCube(cubeOffset, field);
    #pragma omp critical
    totalTriangles += emitedTriangles;
  }





  /*
  for (int i = 0; i < totalCubesCount; i++)
  {
    for (auto it = array_of_coordinates[i].begin(); it != array_of_coordinates[i].end(); it++)
    {
      auto c_back_transformation = (*it / mGridResolution) + (1/2);
      *it = c_back_transformation;
    }

    Vec3_t<float> cubeOffset( array_of_coordinates[i][0],
                             array_of_coordinates[i][1],
                              array_of_coordinates[i][2]);

    unsigned emitedTriangles = buildCube(cubeOffset, field);
    '''

    //#pragma omp critical
    totalTriangles += emitedTriangles;

  }


/*
  // 2. Loop over each coordinate in the 3D grid.
  #pragma omp parallel for schedule(guided)
  for(size_t i = 0; i < totalCubesCount; ++i)
  {
      // 3. Compute 3D position in the grid.
      Vec3_t<float> cubeOffset( i % mGridSize,
                               (i / mGridSize) % mGridSize,
                                i / (mGridSize*mGridSize));

      // 4. Evaluate "Marching Cube" at given position in the grid and
      //    store the number of triangles generated.


      unsigned emitedTriangles = buildCube(cubeOffset, field);
      #pragma omp critical
      totalTriangles += emitedTriangles;
*/

  return totalTriangles;
}

float CachedMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
  // NOTE: This method is called from "buildCube(...)"!

  // 1. Store pointer to and number of 3D points in the field
  //    (to avoid "data()" and "size()" call in the loop).
  const Vec3_t<float> *pPoints = field.getPoints().data();
  const unsigned count = unsigned(field.getPoints().size());

  float value = std::numeric_limits<float>::max();

  // 2. Find minimum square distance from points "pos" to any point in the
  //    field.
  //#pragma omp parallel for
  for(unsigned i = 0; i < count; ++i)
  {
      float distanceSquared  = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x);
      distanceSquared       += (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y);
      distanceSquared       += (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

      // Comparing squares instead of real distance to avoid unnecessary
      // "sqrt"s in the loop.
      value = std::min(value, distanceSquared);
  }

  // 3. Finally take square root of the minimal square distance to get the real distance
  return sqrt(value);
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
