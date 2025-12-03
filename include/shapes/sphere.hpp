#pragma once
#include <device_launch_parameters.h>
void fibonacciSphere(float *p_positions, int p_vertexCount);

//this icosphere method
//face count based on subdivision	| f(sd) = (sd+1)^2 
//recusrively						| f(sd) = f(sd - 1) + 2sd + 1
//vertex count based on subdivision | v(sd) = (sd^2 + 5sd + 6) / 2
//recursively						| v(sd) = v(sd - 1) + sd + 2
void icosphere(float* p_positions, int p_subdivisions); //subdivisions refers to the number of midpoints along the base ico sphere edges
//this function needs to create base octahedron sphere positions,
//add p_subdivision mid points along each edge
//add mid points along the face to be able to retrianglulate face
//project points back to sphere
//trace points to form a EBO to have faces