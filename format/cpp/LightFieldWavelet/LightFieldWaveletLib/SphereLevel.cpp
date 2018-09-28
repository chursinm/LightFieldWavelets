#include "stdafx.h"
#include "SphereLevel.h"
#include <math.h>
#include <algorithm>

namespace LightField
{
	SphereLevel::SphereLevel(SphereLevel* prevLevelIn) :
		prevLevel(prevLevelIn),
		index(prevLevel->index + 1)
	{
		
		vertices.resize	(10 *	(int)pow(4, index)		+ 2	);
		edges.resize	(30 *	(int)pow(4, index)			);
		faces.resize	(20 *	(int)pow(4, index)			);
		//fill indexes
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


		
		perfomSubdivision();
		fillConnectionsGraph();
		

	}
	


	SphereLevel::SphereLevel():index(0)
	{
		vertices.resize(12);
		edges.resize(30);
		faces.resize(20);

		//fill indexes
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

		fillConnectionsGraph();
	}

	void SphereLevel::fillConnectionsGraph()
	{
		std::vector<SphereVertex*>  i, j;
		std::vector<SphereFace*>	facesOfVertices;

		// fill all vertices as begin and end of edges
		for (auto& face : faces)
		{
			i.push_back(face.vertices[0]);
			j.push_back(face.vertices[1]);

			facesOfVertices.push_back(&face);
			
		}
		for (auto& face : faces)
		{
			i.push_back(face.vertices[1]);
			j.push_back(face.vertices[2]);

			facesOfVertices.push_back(&face);
			

		}
		for (auto& face : faces)
		{
			i.push_back(face.vertices[2]);
			j.push_back(face.vertices[0]);

			facesOfVertices.push_back(&face);
			

		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[1]);
			j.push_back(face.vertices[0]);

			facesOfVertices.push_back(&face);
			

		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[2]);
			j.push_back(face.vertices[1]);

			facesOfVertices.push_back(&face);
			

		}

		for (auto& face : faces)
		{
			i.push_back(face.vertices[0]);
			j.push_back(face.vertices[2]);

			facesOfVertices.push_back(&face);
			

		}

		std::vector<int> I;
		std::vector<SphereFace*> facesOfVertices_ordered, facesOfVertices_ordered_inv;
		// take only in ascending direction
		for (int iter = 0; iter < i.size(); iter++)
		{
	
			if (i[iter]->index < j[iter]->index)
			{
				I.push_back(iter);
				facesOfVertices_ordered.push_back(facesOfVertices[iter]);

				
			}
			else
			{
				//facesOfVertices_ordered_inv.push_back(facesOfVertices[iter]);
			}
		}

		//sort them according to bedin and end

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
		std::vector<int> I_inv;
		for (int iter = 0; iter < index.size() / 2; iter++)
		{
			I.push_back(std::max(index[iter * 2 + 1], index[iter * 2]));
			I_inv.push_back(std::min(index[iter * 2 + 1], index[iter * 2]));
		}




		for (int iter = 0; iter < I.size(); iter++)
		{
			edges[iter].vertices[0] = i_ordered[I[iter]];
			edges[iter].vertices[1] = j_ordered[I[iter]];

	
			
			edges[iter].faces[0] = facesOfVertices_ordered[I[iter]];
			edges[iter].faces[1] = facesOfVertices_ordered[I_inv[iter]];



			for (int ind = 0; ind < 6; ind++)
			{
				if (i_ordered[I[iter]]->edges[ind] == nullptr)
				{
					i_ordered[I[iter]]->edges[ind] = &(edges[iter]);
					break;
				}
			}

			for (int ind = 0; ind < 6; ind++)
			{
				if (j_ordered[I[iter]] ->edges[ind]== nullptr)
				{
					j_ordered[I[iter]]->edges[ind] = &(edges[iter]);
					break;
				}
			}


			

			for (int ind = 0; ind < 3; ind++)
			{
				if (facesOfVertices_ordered[I[iter]]->edges[ind] == nullptr)
				{
					facesOfVertices_ordered[I[iter]]->edges[ind] = &(edges[iter]);
					break;
				}
			}

			
			for (int ind = 0; ind < 3; ind++)
			{
				if (facesOfVertices_ordered[I_inv[iter]]->edges[ind] == nullptr)
				{
					facesOfVertices_ordered[I_inv[iter]]->edges[ind] = &(edges[iter]);
					break;
				}
			}

			for (int ind = 0; ind < 6; ind++)
			{
		
				if (i_ordered[I[iter]]->faces[ind] == nullptr)
				{
					i_ordered[I[iter]]->faces[ind] =  facesOfVertices_ordered[I[iter]];
					break;
				}
			}
			for (int ind = 0; ind < 6; ind++)
			{
				if (j_ordered[I[iter]]->faces[ind] == nullptr)
				{
					j_ordered[I[iter]]->faces[ind] = facesOfVertices_ordered[I_inv[iter]];					
					break;
				}
			}

		}	

		for (auto& v : vertices)
		{
			//sort edges			
			std::sort(v.edges, v.edges+6,
				[](const SphereEdge* a, const SphereEdge* b)->bool
			{
				if (a == nullptr) return false;
				if (b == nullptr) return true;
				return a->index<b->index;
			}
			);
			//sort faces
			std::sort(v.faces, v.faces + 6,
				[](const SphereFace* a, const SphereFace* b)->bool
			{
				if (a == nullptr) return false;
				if (b == nullptr) return true;
				return a->index<b->index;
			}
			);						
		}
		for (auto& e : edges)
		{		
			std::sort(e.faces, e.faces + 2,
				[](const SphereFace* a, const SphereFace* b)->bool
			{
				if (a == nullptr) return false;
				if (b == nullptr) return true;
				return a->index<b->index;
			}
			);
			std::sort(e.vertices, e.vertices + 2,
				[](const SphereVertex* a, const SphereVertex* b)->bool
			{
				if (a == nullptr) return false;
				if (b == nullptr) return true;
				return a->index<b->index;
			}
			);

		}
		for (auto& f:faces)
		{
			/*std::sort(f.edges, f.edges + 3,
				[](const SphereEdge* a, const SphereEdge* b)->bool
			{
				if (a == nullptr) return false;
				if (b == nullptr) return true;
				return a->index<b->index;
			}
			);*/
			SphereEdge* edgesTmp[3];
			//for (int i = 0; i < 3; i++) edgesTmp[i] = f.edges[i];
			for (int i = 0; i < 3; i++)
			{
				for (int j = 0; j < 3; j++)
				{
					if (((f.vertices[i]->index == f.edges[j]->vertices[0]->index)&& (f.vertices[(i+1)%3]->index == f.edges[j]->vertices[1]->index))
						||
						(((f.vertices[i]->index == f.edges[j]->vertices[1]->index) && (f.vertices[(i + 1) % 3]->index == f.edges[j]->vertices[0]->index))))
					{
						edgesTmp[i] = f.edges[j];
						break;
					}
				}				
			}
			for (int i = 0; i < 3; i++) { f.edges[i] = edgesTmp[i]; }


		}
		return;
	}

	

	void SphereLevel::perfomSubdivision()
	{
		std::vector<int> v12, v23, v31;

		/*for (const auto& f : prevLevel->faces)
		{			
			v12.push_back(f.edges[0]->index+prevLevel->getNumberOfVertices()+1);
			v23.push_back(f.edges[1]->index+ prevLevel->getNumberOfVertices()+1);
			v31.push_back(f.edges[2]->index+ prevLevel->getNumberOfVertices()+1);
		}*/

		int iter = 0;
		for (auto& f : prevLevel->faces)
		{	

			faces[iter].vertices[0] = &(vertices[(f.vertices[0]->index)]);			
			faces[iter].vertices[1] = &(vertices[f.edges[0]->index + prevLevel->getNumberOfVertices() ]); //v12
			faces[iter].vertices[2] = &(vertices[f.edges[2]->index + prevLevel->getNumberOfVertices() ]); //v31
			f.childFaces[0] = &(faces[iter]);
			faces[iter].parentFace = &f;

			faces[iter + prevLevel->getNumberOfFaces()].vertices[0] = &(vertices[(f.vertices[1]->index)]);
			faces[iter + prevLevel->getNumberOfFaces()].vertices[1] = &(vertices[f.edges[1]->index + prevLevel->getNumberOfVertices()]); //v23
			faces[iter + prevLevel->getNumberOfFaces()].vertices[2] = &(vertices[f.edges[0]->index + prevLevel->getNumberOfVertices()]); //v12
			f.childFaces[1] = &(faces[iter + prevLevel->getNumberOfFaces()]);
			faces[iter + prevLevel->getNumberOfFaces()].parentFace = &f;

			faces[iter + 2 * prevLevel->getNumberOfFaces()].vertices[0] = &(vertices[(f.vertices[2]->index)]);
			faces[iter + 2*prevLevel->getNumberOfFaces()].vertices[1] = &(vertices[f.edges[2]->index + prevLevel->getNumberOfVertices()]); //v31
			faces[iter + 2*prevLevel->getNumberOfFaces()].vertices[2] = &(vertices[f.edges[1]->index + prevLevel->getNumberOfVertices()]); //v23
			f.childFaces[2] = &(faces[iter + 2*prevLevel->getNumberOfFaces()]);
			faces[iter + 2*prevLevel->getNumberOfFaces()].parentFace = &f;

			faces[iter + 3 * prevLevel->getNumberOfFaces()].vertices[0] = &(vertices[f.edges[0]->index + prevLevel->getNumberOfVertices()]); //v12
			faces[iter + 3 * prevLevel->getNumberOfFaces()].vertices[1] = &(vertices[f.edges[1]->index + prevLevel->getNumberOfVertices()]); //v31
			faces[iter + 3 * prevLevel->getNumberOfFaces()].vertices[2] = &(vertices[f.edges[2]->index + prevLevel->getNumberOfVertices()]); //v23
			f.childFaces[3] = &(faces[iter + 3 * prevLevel->getNumberOfFaces()]);
			faces[iter + 3 * prevLevel->getNumberOfFaces()].parentFace = &f;

			f.edges[0]->childVertex = &(vertices[f.edges[0]->index + prevLevel->getNumberOfVertices() ]);
			f.edges[1]->childVertex = &(vertices[f.edges[1]->index + prevLevel->getNumberOfVertices() ]);
			f.edges[2]->childVertex = &(vertices[f.edges[2]->index + prevLevel->getNumberOfVertices() ]);

			vertices[f.edges[0]->index + prevLevel->getNumberOfVertices()].parentEdge = f.edges[0];
			vertices[f.edges[1]->index + prevLevel->getNumberOfVertices()].parentEdge = f.edges[1];
			vertices[f.edges[2]->index + prevLevel->getNumberOfVertices()].parentEdge = f.edges[2];
			iter++;
		}

		for (int ind = 0; ind < prevLevel->getNumberOfVertices(); ind++)
		{
			vertices[ind] = prevLevel->vertices[ind];
			vertices[ind].directParentVertex = &(prevLevel->vertices[ind]);
		}

		for (int ind = prevLevel->getNumberOfVertices(); ind < getNumberOfVertices(); ind++)
		{
			vertices[ind].parentEdge = &(prevLevel->edges[ind- prevLevel->getNumberOfVertices()]);
			prevLevel->edges[ind - prevLevel->getNumberOfVertices()].childVertex = &(vertices[ind]);
			vertices[ind].pos = glm::normalize((	vertices[ind].parentEdge->vertices[0]->pos +
													vertices[ind].parentEdge->vertices[1]->pos) 
													/ 2.0f);
								
		}

		return;
	}

	

}


