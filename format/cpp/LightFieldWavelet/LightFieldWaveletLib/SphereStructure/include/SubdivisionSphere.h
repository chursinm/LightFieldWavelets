#pragma once
#include "Constants.h"
#include <vector>
#include <algorithm>
#include <map>
#include <iostream>

// Subdivision sphere structure

namespace SubdivisionShpere {

	struct Vertex
	{
		float x = 0.0f;
		float y = 0.0f;
		float z = 0.0f;
		int creationLevel = 0;
		int parentA = 0;
		int parentB = 0;
	};

	struct Face
	{
		int vertA = 0;
		int vertB = 0;
		int vertC = 0;
		int childFaceA = 0;
		int childFaceB = 0;
		int childFaceC = 0;
		int childFaceD = 0;
		int parentFace = 0;
	};

	struct Level
	{
		Vertex* vertices = nullptr;
		Face* faces = nullptr;
		int numberOfVertices = 0;
		int numberOfFaces = 0;
		int levelIndex = 0;
	};

	class SubdivisionSphere
	{
	public:
		SubdivisionSphere();
		SubdivisionSphere(int numberOfLevels);
		~SubdivisionSphere();
		int getNumberOfLevels() { return _numberOfLevels; }
		Level getLevel(int levelIndex) { return levels[levelIndex]; }

	private:
		void initLevel();
		Level* levels = nullptr;
		int _numberOfLevels;
		bool performSubdivision(Level* prevLevel, Level* nextLevel);

	};



}




