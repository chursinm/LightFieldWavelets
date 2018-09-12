#include "stdafx.h"
#include "SphereInitialLevel.h"

namespace LightField 
{
	LightField::SphereInitialLevel::SphereInitialLevel()
	{
		vertices.resize(12);
		edges.resize(30);
		faces.resize(20);

		for (int i = 0; i < vertices.size(); i++)
		{
			vertices[i].index = i;
		}
		for (int i = 0; i < edges.size(); i++)
		{
			edges[i].index = i;
		}
		for (int i = 0; i < faces.size(); i++)
		{
			faces[i].index = i;
		}


		float tau = 0.8506508084f;
		float one = 0.5257311121f;

		//vertices
		vertices[0].pos.x =		 tau;			vertices[0].pos.y =		 one;			vertices[0].pos.z =		  0;
		vertices[1].pos.x =		-tau;			vertices[1].pos.y =		 one;			vertices[1].pos.z =		  0;
		vertices[2].pos.x =		-tau;			vertices[2].pos.y =		-one;			vertices[2].pos.z =		  0;
		vertices[3].pos.x =		 tau;			vertices[3].pos.y =		-one;			vertices[3].pos.z =		  0;
		vertices[4].pos.x =		 one;			vertices[4].pos.y =		   0;			vertices[4].pos.z =		tau;

		vertices[5].pos.x =		 one;			vertices[5].pos.y =			0;			vertices[5].pos.z =		-tau;
		vertices[6].pos.x =		-one;			vertices[6].pos.y =			0;			vertices[6].pos.z =		-tau;
		vertices[7].pos.x =		-one;			vertices[7].pos.y =			0;			vertices[7].pos.z =	  	 tau;
		vertices[8].pos.x =		   0;			vertices[8].pos.y =		  tau;			vertices[8].pos.z =		 one;
		vertices[9].pos.x =		   0;			vertices[9].pos.y =		 -tau;			vertices[9].pos.z =		 one;

		vertices[10].pos.x =		0;			vertices[10].pos.y =	 -tau;			vertices[10].pos.z =-	 one;
		vertices[11].pos.x =		0;			vertices[11].pos.y =	  tau;			vertices[11].pos.z =	-one;


		//faces

		faces[0].childFaces[0] = &(faces[4]);		faces[0].childFaces[1] = &(faces[8]);		faces[0].childFaces[2] = &(faces[7]);
		faces[1].childFaces[0] = &(faces[4]);		faces[1].childFaces[1] = &(faces[7]);		faces[1].childFaces[2] = &(faces[9]);
		faces[2].childFaces[0] = &(faces[5]);		faces[2].childFaces[1] = &(faces[6]);		faces[2].childFaces[2] = &(faces[11]);
		faces[3].childFaces[0] = &(faces[5]);		faces[3].childFaces[1] = &(faces[10]);		faces[3].childFaces[2] = &(faces[6]);
		faces[4].childFaces[0] = &(faces[0]);		faces[4].childFaces[1] = &(faces[4]);		faces[4].childFaces[2] = &(faces[3]);

		faces[5].childFaces[0] = &(faces[0]);		faces[5].childFaces[1] = &(faces[3]);		faces[5].childFaces[2] = &(faces[5]);
		faces[6].childFaces[0] = &(faces[2]);		faces[6].childFaces[1] = &(faces[7]);		faces[6].childFaces[2] = &(faces[1]);
		faces[7].childFaces[0] = &(faces[2]);		faces[7].childFaces[1] = &(faces[1]);		faces[7].childFaces[2] = &(faces[6]);
		faces[8].childFaces[0] = &(faces[8]);		faces[8].childFaces[1] = &(faces[0]);		faces[8].childFaces[2] = &(faces[11]);
		faces[9].childFaces[0] = &(faces[8]);		faces[9].childFaces[1] = &(faces[11]);		faces[9].childFaces[2] = &(faces[1]);

		faces[10].childFaces[0] = &(faces[9]);		faces[10].childFaces[1] = &(faces[10]);		faces[10].childFaces[2] = &(faces[3]);
		faces[11].childFaces[0] = &(faces[9]);		faces[11].childFaces[1] = &(faces[2]);		faces[11].childFaces[2] = &(faces[10]);
		faces[12].childFaces[0] = &(faces[8]);		faces[12].childFaces[1] = &(faces[4]);		faces[12].childFaces[2] = &(faces[0]);
		faces[13].childFaces[0] = &(faces[11]);		faces[13].childFaces[1] = &(faces[0]);		faces[13].childFaces[2] = &(faces[5]);
		faces[14].childFaces[0] = &(faces[4]);		faces[14].childFaces[1] = &(faces[9]);		faces[14].childFaces[2] = &(faces[3]);

		faces[15].childFaces[0] = &(faces[5]);		faces[15].childFaces[1] = &(faces[3]);		faces[15].childFaces[2] = &(faces[10]);
		faces[16].childFaces[0] = &(faces[7]);		faces[16].childFaces[1] = &(faces[8]);		faces[16].childFaces[2] = &(faces[1]);
		faces[17].childFaces[0] = &(faces[6]);		faces[17].childFaces[1] = &(faces[1]);		faces[17].childFaces[2] = &(faces[11]);
		faces[18].childFaces[0] = &(faces[7]);		faces[18].childFaces[1] = &(faces[2]);		faces[18].childFaces[2] = &(faces[9]);
		faces[19].childFaces[0] = &(faces[6]);		faces[19].childFaces[1] = &(faces[10]);		faces[19].childFaces[2] = &(faces[2]);

		
	}

}

