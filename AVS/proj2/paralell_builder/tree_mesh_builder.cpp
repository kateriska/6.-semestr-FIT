/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Kateruna Fortova <xforto00@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    16.12.2020
 **/

#include <iostream>
#include <math.h>
#include <limits>

#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{

}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.

    unsigned count = 0;

    #pragma omp parallel shared(count)
    {
      #pragma omp single
      count = octaTree(field, mGridSize, Vec3_t<float>());
    }


    return count;
}

unsigned TreeMeshBuilder::octaTree(const ParametricScalarField &field, unsigned mGridSize, const Vec3_t<float> &cubeOffset)
{
  unsigned count = 0;

  float half_block_length = (mGridSize * mGridResolution) / 2.0;

  // compute midpoint of block
  const Vec3_t<float> block_midpoint(
		half_block_length + cubeOffset.x * mGridResolution,
		half_block_length + cubeOffset.y * mGridResolution,
		half_block_length + cubeOffset.z * mGridResolution);

  // check for every child equation 6.3 (children should not be empty)
  float F_p = evaluateFieldAt(block_midpoint, field);

  float a = mGridSize * mGridResolution;
  float check_empty_child_expression = mIsoLevel + (sqrt(3.0) / 2.0) * a;

  if (F_p > check_empty_child_expression)
  {
    return 0;
  }

  if (mGridSize <= 1)
	{
    // finally call build cube on lowest level
    unsigned emited_triangles = 0;

    emited_triangles = buildCube(cubeOffset, field);

		return emited_triangles;
	}

  // compute new smaller size of grid
  mGridSize = mGridSize / 2.0;

  for (const Vec3_t<float> v: sc_vertexNormPos)
	{
    #pragma omp task firstprivate(v) shared(count)
    {
      // separate children to new children and recursivelly call octaTree
      const Vec3_t<float> new_cube_child(
				v.x * mGridSize + cubeOffset.x,
				v.y * mGridSize + cubeOffset.y,
				v.z * mGridSize + cubeOffset.z);

      // compute count of triangles for this iteration
      int iteration_count =  octaTree(field, mGridSize, new_cube_child);

      #pragma omp atomic
      count = count + iteration_count; // compute final count of triangles
    }
  }

  #pragma omp taskwait
  return count;


}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
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

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
  // NOTE: This method is called from "buildCube(...)"!

  // Store generated triangle into vector (array) of generated triangles.
  // The pointer to data in this array is return by "getTrianglesArray(...)" call
  // after "marchCubes(...)" call ends.
  #pragma omp critical
  mTriangles.push_back(triangle);

}
