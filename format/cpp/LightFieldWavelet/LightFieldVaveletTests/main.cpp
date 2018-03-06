#include <iostream>
#include <SphereStructure\include\SubdivisionSphere.h>
#include <glm/glm.hpp>
#include <glm/gtc/random.hpp>

using namespace SubdivisionShpere;



int main()
{
	std::cout << "light field wavelet format cpp start" << std::endl;	
	SubdivisionSphere* subsphere = new SubdivisionSphere(9);

	//glm::vec3 testVect = subsphere->indexToVector(0, 0);
	//int index = subsphere->vectorToIndex(testVect, 0);

	
	//std::vector <Face> faceswith2765;
	//std::vector <int> indexOffaceswith2765;

	//auto res = subsphere->findAllAjustmentFaces(2765, 5);
	//glm::vec3 testVect1 = subsphere->indexToVector(2765, 5);
	//int index1 = subsphere->vectorToFaceIndex(testVect1, 5);

	//test with triangles






	for (int iter = 0; iter < subsphere->getNumberOfLevels(); iter++)
	{
		std::cout << "check level " << iter << std::endl;
		for (int iter1 = 0; iter1 < subsphere->getLevel(iter).numberOfFaces; iter1++)
		{
			for (int testSample = 0; testSample < 10; testSample++)
			{

				glm::vec3 sample = glm::linearRand(glm::vec3(EPSILON), glm::vec3(1- EPSILON));
				glm::vec3 baricetric = sample / (sample.x + sample.y + sample.z);
				Face* testFace = &(subsphere->getLevel(iter).faces[iter1]);
				glm::vec3 testVector =			testFace->vertARef->position*baricetric.x +
												testFace->vertBRef->position*baricetric.y +
												testFace->vertCRef->position*baricetric.z;
				int indexOfFace = subsphere->vectorToFaceIndex(testVector, iter);
				if (indexOfFace != iter1)
				{
					std::cout << "error in face detection" << std::endl;
					indexOfFace = subsphere->vectorToFaceIndex(testVector, iter);
					std::cout << "indexOfFace "<< indexOfFace << std::endl;
				}
			}
		}
	}

	/*for (int iter = 0; iter <6 ; iter++)
	{
		
		for (int iter1 = 0; iter1 < subsphere->getLevel(iter).numberOfVertices; iter1++)
		{
			if (iter == 5 && iter1 == 2765)
			{
				std::cout << "stop" << std::endl;
			}
			glm::vec3 testVect = subsphere->indexToVector(iter1, iter);
			//testVect.x += 0.0001;
			//testVect = glm::normalize(testVect);
			//int index = subsphere->vectorToIndex(testVect, iter);
			//if (index != iter1)
			//{
			//	std::cout << "index = " << index << std::endl;
			//}
			
		}
	}*/
	
	return 0;
}