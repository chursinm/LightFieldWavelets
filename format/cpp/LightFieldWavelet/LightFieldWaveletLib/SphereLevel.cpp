#include "stdafx.h"
#include "SphereLevel.h"
#include <math.h>
#include <algorithm>

namespace LightField
{
	SphereLevel::SphereLevel(SphereLevel* prevLevel):index(prevLevel->index+1)
	{
		vertices.resize	(10 *	(int)pow(4, index)		+ 2);
		edges.resize	(30 *	(int)pow(4, index)		);
		faces.resize	(20 *	(int)pow(4, index)		);

	}
	


	SphereLevel::SphereLevel():index(0)
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
		vertices[0].pos.x = tau;			vertices[0].pos.y = one;			vertices[0].pos.z = 0;
		vertices[1].pos.x = -tau;			vertices[1].pos.y = one;			vertices[1].pos.z = 0;
		vertices[2].pos.x = -tau;			vertices[2].pos.y = -one;			vertices[2].pos.z = 0;
		vertices[3].pos.x = tau;			vertices[3].pos.y = -one;			vertices[3].pos.z = 0;
		vertices[4].pos.x = one;			vertices[4].pos.y = 0;			vertices[4].pos.z = tau;

		vertices[5].pos.x = one;			vertices[5].pos.y = 0;			vertices[5].pos.z = -tau;
		vertices[6].pos.x = -one;			vertices[6].pos.y = 0;			vertices[6].pos.z = -tau;
		vertices[7].pos.x = -one;			vertices[7].pos.y = 0;			vertices[7].pos.z = tau;
		vertices[8].pos.x = 0;			vertices[8].pos.y = tau;			vertices[8].pos.z = one;
		vertices[9].pos.x = 0;			vertices[9].pos.y = -tau;			vertices[9].pos.z = one;

		vertices[10].pos.x = 0;			vertices[10].pos.y = -tau;			vertices[10].pos.z = -one;
		vertices[11].pos.x = 0;			vertices[11].pos.y = tau;			vertices[11].pos.z = -one;


		//faces

		faces[0].vertices[0] = &(vertices[4]);		faces[0].vertices[1] = &(vertices[8]);		faces[0].vertices[2] = &(vertices[7]);
		faces[1].vertices[0] = &(vertices[4]);		faces[1].vertices[1] = &(vertices[7]);		faces[1].vertices[2] = &(vertices[9]);
		faces[2].vertices[0] = &(vertices[5]);		faces[2].vertices[1] = &(vertices[6]);		faces[2].vertices[2] = &(vertices[11]);
		faces[3].vertices[0] = &(vertices[5]);		faces[3].vertices[1] = &(vertices[10]);		faces[3].vertices[2] = &(vertices[6]);
		faces[4].vertices[0] = &(vertices[0]);		faces[4].vertices[1] = &(vertices[4]);		faces[4].vertices[2] = &(vertices[3]);

		faces[5].vertices[0] = &(vertices[0]);		faces[5].vertices[1] = &(vertices[3]);		faces[5].vertices[2] = &(vertices[5]);
		faces[6].vertices[0] = &(vertices[2]);		faces[6].vertices[1] = &(vertices[7]);		faces[6].vertices[2] = &(vertices[1]);
		faces[7].vertices[0] = &(vertices[2]);		faces[7].vertices[1] = &(vertices[1]);		faces[7].vertices[2] = &(vertices[6]);
		faces[8].vertices[0] = &(vertices[8]);		faces[8].vertices[1] = &(vertices[0]);		faces[8].vertices[2] = &(vertices[11]);
		faces[9].vertices[0] = &(vertices[8]);		faces[9].vertices[1] = &(vertices[11]);		faces[9].vertices[2] = &(vertices[1]);

		faces[10].vertices[0] = &(vertices[9]);		faces[10].vertices[1] = &(vertices[10]);	faces[10].vertices[2] = &(vertices[3]);
		faces[11].vertices[0] = &(vertices[9]);		faces[11].vertices[1] = &(vertices[2]);		faces[11].vertices[2] = &(vertices[10]);
		faces[12].vertices[0] = &(vertices[8]);		faces[12].vertices[1] = &(vertices[4]);		faces[12].vertices[2] = &(vertices[0]);
		faces[13].vertices[0] = &(vertices[11]);	faces[13].vertices[1] = &(vertices[0]);		faces[13].vertices[2] = &(vertices[5]);
		faces[14].vertices[0] = &(vertices[4]);		faces[14].vertices[1] = &(vertices[9]);		faces[14].vertices[2] = &(vertices[3]);

		faces[15].vertices[0] = &(vertices[5]);		faces[15].vertices[1] = &(vertices[3]);		faces[15].vertices[2] = &(vertices[10]);
		faces[16].vertices[0] = &(vertices[7]);		faces[16].vertices[1] = &(vertices[8]);		faces[16].vertices[2] = &(vertices[1]);
		faces[17].vertices[0] = &(vertices[6]);		faces[17].vertices[1] = &(vertices[1]);		faces[17].vertices[2] = &(vertices[11]);
		faces[18].vertices[0] = &(vertices[7]);		faces[18].vertices[1] = &(vertices[2]);		faces[18].vertices[2] = &(vertices[9]);
		faces[19].vertices[0] = &(vertices[6]);		faces[19].vertices[1] = &(vertices[10]);	faces[19].vertices[2] = &(vertices[2]);

		initilizeEdges();
	}

	void SphereLevel::initilizeEdges()
	{
		std::vector<SphereVertex*>  i, j;
		for (auto& face : faces)
		{
			i.push_back(face.vertices[0]);
			j.push_back(face.vertices[1]);
		}
		for (auto& face : faces)
		{
			i.push_back(face.vertices[1]);
			j.push_back(face.vertices[2]);
		}
		for (auto& face : faces)
		{
			i.push_back(face.vertices[2]);
			j.push_back(face.vertices[0]);
		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[1]);
			j.push_back(face.vertices[0]);
		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[2]);
			j.push_back(face.vertices[1]);
		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[0]);
			j.push_back(face.vertices[2]);
		}

		std::vector<int> I;

		for (int iter = 0; iter < i.size(); iter++)
		{
	
			if (i[iter]->index < j[iter]->index)
			{
				I.push_back(iter);
			}
		}
		std::vector<SphereVertex*>  i_ordered, j_ordered;

		for (auto index : I)
		{
			i_ordered.push_back(i[index]);
			j_ordered.push_back(j[index]);
		}

		std::vector<int> index;

		for (int iter = 0; iter < i_ordered.size(); iter++)
		{
			index.push_back(iter);
		}


		

		std::sort(index.begin(), index.end(),
					[&](const int& a, const int& b)
					{			
						if (j_ordered[a]->index == j_ordered[b]->index)
						{
							return (i_ordered[a]->index < i_ordered[b]->index);
						}
						else
						{
							return (j_ordered[a]->index < j_ordered[b]->index);
						}
					}
				);
		I.clear();

		for (int iter = 0; iter < index.size() / 2; iter++)
		{
			I.push_back(std::max(index[iter * 2 + 1], index[iter * 2]));
		}

		for (int iter = 0; iter < I.size(); iter++)
		{
			edges[iter].vertices[0] = i_ordered[I[iter]];
			edges[iter].vertices[1] = j_ordered[I[iter]];
		}
	}



}


