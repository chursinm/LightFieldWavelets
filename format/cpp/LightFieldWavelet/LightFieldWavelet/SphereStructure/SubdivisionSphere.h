#pragma once
#include "Constants.h"
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>

// Subdivision sphere structure

struct Vertex
{
	float x=0.0f;
	float y=0.0f;
	float z=0.0f;
};

struct Face
{
	int vertA=0;
	int vertB=0;
	int vertC=0;
};

struct Level
{
	Vertex* vertices = nullptr;
	Face* faces = nullptr;
	int numberOfVertices = 0;
	int numberOfFaces = 0;
};

class SubdivisionSphere
{
public:
	SubdivisionSphere();
	SubdivisionSphere(int numberOfLevels);
	~SubdivisionSphere();
	int getNumberOfLevels() { return _numberOfLevels; }

private:
	void initLevel();
	Level* levels= nullptr;
	int _numberOfLevels;
	bool SubdivisionSphere::performSubdivision(Level* prevLevel, Level* nextLevel);

};




