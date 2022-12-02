/**
 * @file    tree_mesh_builder.cpp
 *
 * @author  Josef Kotoun <xkotou06@stud.fit.vutbr.cz>
 *
 * @brief   Parallel Marching Cubes implementation using OpenMP tasks + octree early elimination
 *
 * @date    2.12.2022
 **/

#include <iostream>
#include <math.h>
#include <limits>
#include <omp.h>
#include "tree_mesh_builder.h"

TreeMeshBuilder::TreeMeshBuilder(unsigned gridEdgeSize)
    : BaseMeshBuilder(gridEdgeSize, "Octree")
{
}

unsigned TreeMeshBuilder::splitCube(Vec3_t<float> &cubePosition, const ParametricScalarField field, int edgeLen)
{
    int min_edge_size = 2;
    unsigned totalTriangles = 0;
    // splitting limit - march cubes
    if (edgeLen <= min_edge_size)
    {
    
        int totalCubesCount = edgeLen * edgeLen * edgeLen;
        #pragma omp parallel for reduction(+:totalTriangles) schedule(guided)
        for (size_t i = 0; i < totalCubesCount; ++i)
        {
            Vec3_t<float> cubeOffset(cubePosition.x + (i % edgeLen),
                                     cubePosition.y + ((i / edgeLen) % edgeLen),
                                     cubePosition.z + (i / (edgeLen * edgeLen)));

            totalTriangles += buildCube(cubeOffset, field);
        }
        return totalTriangles;
    }

    int subCubeEdgeLen = edgeLen / 2;
    for (int i = 0; i < 8; i++)
    {
        // left bottom vertex of each cube
        Vec3_t<float> subCubePos = {cubePosition.x + sc_vertexNormPos[i].x * subCubeEdgeLen, cubePosition.y + sc_vertexNormPos[i].y * subCubeEdgeLen, cubePosition.z + sc_vertexNormPos[i].z * subCubeEdgeLen};
        Vec3_t<float> cubeMidPosTranformed = {(subCubePos.x + (subCubeEdgeLen / 2)) * mGridResolution, (subCubePos.y + (subCubeEdgeLen / 2)) * mGridResolution, (subCubePos.z + (subCubeEdgeLen / 2)) * mGridResolution};
      
        double fieldVal = evaluateFieldAt(cubeMidPosTranformed, field);
        double emptyThresh = mIsoLevel + ((sqrt(3) / 2) * (subCubeEdgeLen * mGridResolution));
        // cube is not empty
        if ((fieldVal <= emptyThresh))
        {
            #pragma omp task default(none) shared(field, totalTriangles) firstprivate(subCubePos, subCubeEdgeLen)
            {
                unsigned subCubeTriangles = splitCube(subCubePos, field, subCubeEdgeLen);
                #pragma omp atomic update
                totalTriangles += subCubeTriangles;
            }
        }
    }
#pragma omp taskwait
    return totalTriangles;
}

unsigned TreeMeshBuilder::marchCubes(const ParametricScalarField &field)
{
    // Suggested approach to tackle this problem is to add new method to
    // this class. This method will call itself to process the children.
    // It is also strongly suggested to first implement Octree as sequential
    // code and only when that works add OpenMP tasks to achieve parallelism.
    Vec3_t<float> CubePos = {0, 0, 0};
    unsigned totalTriangles = 0;
// split cubes

#pragma omp parallel shared(field, totalTriangles)
#pragma omp master
    totalTriangles = splitCube(CubePos, field, mGridSize);
    return totalTriangles;
}

float TreeMeshBuilder::evaluateFieldAt(const Vec3_t<float> &pos, const ParametricScalarField &field)
{
    const Vec3_t<float> *pPoints = field.getPoints().data();
    const unsigned count = unsigned(field.getPoints().size());

    float value = std::numeric_limits<float>::max();

    for (unsigned i = 0; i < count; ++i)
    {
        float distanceSquared = (pos.x - pPoints[i].x) * (pos.x - pPoints[i].x) +
                                (pos.y - pPoints[i].y) * (pos.y - pPoints[i].y) + (pos.z - pPoints[i].z) * (pos.z - pPoints[i].z);

        value = std::min(value, distanceSquared);
    }
    return sqrt(value);
}

void TreeMeshBuilder::emitTriangle(const BaseMeshBuilder::Triangle_t &triangle)
{
#pragma omp critical(trianglesVector)
    mTriangles.push_back(triangle);
}
