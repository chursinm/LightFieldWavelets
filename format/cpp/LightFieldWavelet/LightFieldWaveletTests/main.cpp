#include <iostream>
#include <vector>
#include <SubdivisionSphere.h>
#include <WaveletCompression.h>
#include <SphereVertex.h>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>
#include <Constants.h>





int main()
{	
	LightField::SubdivisionSphere  subsphere(0);


	std::cout << "light field wavelet format cpp start" << std::endl;
	

	
	

	//test with triangles


	//test for specific face
	/*{
		glm::vec3 testVector = subsphere.getLevel(0).getFace(2).vertices[0]->pos* 0.333333f +
			subsphere.getLevel(0).getFace(2).vertices[1]->pos* 0.333333f +
			subsphere.getLevel(0).getFace(2).vertices[2]->pos* 0.333333f;
		int indexOfFace = subsphere.vectorToFaceIndex(testVector, 0);
	}*/


	
	for (const auto& level : subsphere.getLevels())
	{
		std::cout << "check level " << level.getIndex() << std::endl;
		for (const auto& face : level.getFaces())
		{
			for (int testSample = 0; testSample < 10; testSample++)
			{
				glm::vec3 sample = glm::linearRand(glm::vec3(EPSILON), glm::vec3(1 - EPSILON));
				glm::vec3 baricetric = sample / (sample.x + sample.y + sample.z);
				glm::vec3 testVector =	face.vertices[0]->pos*baricetric.x +
										face.vertices[1]->pos*baricetric.y +
										face.vertices[2]->pos*baricetric.z;
				int indexOfFace = subsphere.vectorToFaceIndex(testVector, level.getIndex());
				if (indexOfFace!=face.index)
				{
					std::cout << "error in face detection" << std::endl;
					indexOfFace = subsphere.vectorToFaceIndex(testVector, level.getIndex());
					std::cout << "indexOfFace " << indexOfFace << std::endl;
				}
					
			}
		}
	}
	return 0;
}