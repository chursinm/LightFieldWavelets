#include <iostream>
#include <vector>
#include <SubdivisionSphere.h>
#include <WaveletCompression.h>
#include <SphereVertex.h>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <Constants.h>


bool testConnectivity(LightField::SubdivisionSphere& subsphere)
{
	for (const auto& level : subsphere.getLevels())
	{
		for (const auto& vert : level.getVertices())
		{
			//check edges to vert
			for (const auto& edge : vert.edges)
			{
				if (edge != nullptr)
				{
					if (!(edge->vertices[0] == &vert) && !(edge->vertices[1] == &vert))
					{
						std::cout << "edge to vert connectivity failed";
						return false;
					}
				}
			}
			//check faces to vert
			for (const auto& face : vert.faces)
			{
				if (face != nullptr)
				{
					bool testFailed = true;
					for (const auto& v : face->vertices)
					{
						if (v == &vert)
						{
							testFailed = false;
							break;
						}
					}
					if (testFailed)
					{
						std::cout << "face to vert connectivity failed";
						return false;
					}
				}
			}
		}
		//check edges
		for (const auto& edge : level.getEdges())
		{
			// check connectivity vert to edges
			for (const auto& v : edge.vertices)
			{
				bool testFailed = true;
				for (const auto& ed : v->edges)
				{
					if (ed == &edge)
					{
						testFailed = false;
						break;
					}
				}
				if (testFailed)
				{
					std::cout << "vertex to edge connectivity failed";
					return false;
				}
			}
			// check connectivity face to edges
			for (const auto& f : edge.faces)
			{
				bool testFailed = true;
				for (const auto& ed : f->edges)
				{
					if (ed == &edge)
					{
						testFailed = false;
						break;
					}
				}
				if (testFailed)
				{
					std::cout << "face to edge connectivity failed" << std::endl;
					return false;
				}
			}
		}
		//check faces
		for (const auto& face : level.getFaces())
		{

			for (const auto& v : face.vertices)
			{
				bool testFailed = true;
				for (const auto& f : v->faces)
				{
					if (f == &face)
					{
						testFailed = false;
						break;
					}
				}
				if (testFailed)
				{
					std::cout << "vertex to face connectivity failed" << std::endl;
					return false;
				}
			}
			for (const auto& e : face.edges)
			{
				bool testFailed = true;
				for (const auto& f : e->faces)
				{
					if (f == &face)
					{
						testFailed = false;
						break;
					}
				}
				if (testFailed)
				{
					std::cout << "edge to face connectivity failed" << std::endl;
					return false;
				}
			}

		}
	}
	return true;
}

bool testTriangles(LightField::SubdivisionSphere& subsphere)
{
	for (const auto& level : subsphere.getLevels())
	{
		std::cout << "check level " << level.getIndex() << std::endl;
		for (const auto& face : level.getFaces())
		{
			for (int testSample = 0; testSample < 10; testSample++)
			{
				glm::vec3 sample = glm::linearRand(glm::vec3(EPSILON), glm::vec3(1 - EPSILON));
				//glm::vec3 sample(0.333, 0.333, 0.333);
				glm::vec3 baricetric = sample / (sample.x + sample.y + sample.z);
				glm::vec3 testVector = face.vertices[0]->pos*baricetric.x +
					face.vertices[1]->pos*baricetric.y +
					face.vertices[2]->pos*baricetric.z;
				int indexOfFace = subsphere.vectorToFaceIndex(testVector, level.getIndex());
				if (indexOfFace != face.index)
				{
					std::cout << "error in face detection" << std::endl;
					indexOfFace = subsphere.vectorToFaceIndex(testVector, level.getIndex());
					std::cout << "indexOfFace " << indexOfFace << std::endl;
					return false;
				}

			}
		}
	}

}



int main()
{	
	LightField::SubdivisionSphere  subsphere(10);

	

	std::cout << "light field wavelet format cpp start" << std::endl;
	
	if (!testConnectivity(subsphere))
	{
		std::cout << "error in geometry structure" << std::endl;
		return 1;
	}

	if (!testTriangles(subsphere))
	{
		std::cout << "error in triangles test" << std::endl;
	}
	
	

	


	//test for specific face
	/*{
		glm::vec3 testVector = subsphere.getLevel(0).getFace(2).vertices[0]->pos* 0.333333f +
			subsphere.getLevel(0).getFace(2).vertices[1]->pos* 0.333333f +
			subsphere.getLevel(0).getFace(2).vertices[2]->pos* 0.333333f;
		int indexOfFace = subsphere.vectorToFaceIndex(testVector, 0);
	}*/


	
	return 0;
}