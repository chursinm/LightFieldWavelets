#include "stdafx.h"
#include <glm/glm.hpp>
//#include <glm/ext.hpp>
#include <SphereStructure\include\SubdivisionSphere.h>


using namespace SubdivisionShpere;

SubdivisionSphere::SubdivisionSphere()
{

}

SubdivisionSphere::~SubdivisionSphere()
{
	for (int i = 0; i < _numberOfLevels; i++)
	{
		delete[] levels[i].vertices;
		delete[] levels[i].faces;
	}
	delete[] levels;
}

SubdivisionSphere::SubdivisionSphere(int numberOfLevels)
{
	_numberOfLevels = numberOfLevels;
	levels = new Level[_numberOfLevels];
	initLevel();
	for (int iter = 1; iter < _numberOfLevels; iter++)
	{
		performSubdivision(&(levels[iter - 1]), &(levels[iter]));
	}
	for (int iter = 0; iter < numberOfLevels; iter++)
	{
		for (int iter1 = 0; iter1 < levels[iter].numberOfFaces; iter1++)
		{
			levels[iter].faces[iter1].vertA--;
			levels[iter].faces[iter1].vertB--;
			levels[iter].faces[iter1].vertC--;
			levels[iter].faces[iter1].index = iter1;

			if (iter < numberOfLevels - 1)
			{
				levels[iter].faces[iter1].childFaceARef = &(levels[iter+1].faces[levels[iter].faces[iter1].childFaceA]);
				levels[iter].faces[iter1].childFaceBRef = &(levels[iter+1].faces[levels[iter].faces[iter1].childFaceB]);
				levels[iter].faces[iter1].childFaceCRef = &(levels[iter+1].faces[levels[iter].faces[iter1].childFaceC]);
				levels[iter].faces[iter1].childFaceDRef = &(levels[iter+1].faces[levels[iter].faces[iter1].childFaceD]);
			}

			if (iter > 0)
			{
				levels[iter].faces[iter1].parentFaceRef = &(levels[iter-1].faces[levels[iter].faces[iter1].parentFace]);
			}

			levels[iter].faces[iter1].vertARef = &(levels[iter].vertices[levels[iter].faces[iter1].vertA]);
			levels[iter].faces[iter1].vertBRef = &(levels[iter].vertices[levels[iter].faces[iter1].vertB]);
			levels[iter].faces[iter1].vertCRef = &(levels[iter].vertices[levels[iter].faces[iter1].vertC]);


		}
		for (int iter1 = 0; iter1 < levels[iter].numberOfVertices; iter1++)
		{
			levels[iter].vertices[iter1].index = iter1;
			if (iter > 0)
			{
				levels[iter].vertices[iter1].parentARef = &(levels[iter - 1].vertices[levels[iter].vertices[iter1].parentA]);
				levels[iter].vertices[iter1].parentBRef = &(levels[iter - 1].vertices[levels[iter].vertices[iter1].parentB]);
			}
		}

	}

}


void SubdivisionSphere::initLevel()
{
	Level& baseLevel = levels[0];
	baseLevel.levelIndex = 0;
	baseLevel.vertices = new Vertex[12];
	baseLevel.numberOfVertices = 12;
	baseLevel.faces = new Face[20];
	baseLevel.numberOfFaces = 20;
	float tau = 0.8506508084f;
	float one = 0.5257311121f;
	//vertices 
	baseLevel.vertices[0].position.x = tau;	baseLevel.vertices[0].position.y = one;    baseLevel.vertices[0].position.z = 0;
	baseLevel.vertices[1].position.x = -tau;	baseLevel.vertices[1].position.y = one;	 baseLevel.vertices[1].position.z = 0;
	baseLevel.vertices[2].position.x = -tau;	baseLevel.vertices[2].position.y = -one;    baseLevel.vertices[2].position.z = 0;
	baseLevel.vertices[3].position.x = tau;	baseLevel.vertices[3].position.y = -one;    baseLevel.vertices[3].position.z = 0;
	baseLevel.vertices[4].position.x = one;	baseLevel.vertices[4].position.y = 0;    baseLevel.vertices[4].position.z = tau;
	baseLevel.vertices[5].position.x = one; 	baseLevel.vertices[5].position.y = 0;    baseLevel.vertices[5].position.z = -tau;
	baseLevel.vertices[6].position.x = -one;	baseLevel.vertices[6].position.y = 0;    baseLevel.vertices[6].position.z = -tau;
	baseLevel.vertices[7].position.x = -one;	baseLevel.vertices[7].position.y = 0;    baseLevel.vertices[7].position.z = tau;
	baseLevel.vertices[8].position.x = 0;	baseLevel.vertices[8].position.y = tau;    baseLevel.vertices[8].position.z = one;
	baseLevel.vertices[9].position.x = 0;	baseLevel.vertices[9].position.y = -tau;    baseLevel.vertices[9].position.z = one;
	baseLevel.vertices[10].position.x = 0;	baseLevel.vertices[10].position.y = -tau;    baseLevel.vertices[10].position.z = -one;
	baseLevel.vertices[11].position.x = 0;	baseLevel.vertices[11].position.y = tau;    baseLevel.vertices[11].position.z = -one;
	//faces

	baseLevel.faces[0].vertA = 5;	baseLevel.faces[0].vertB = 9;	baseLevel.faces[0].vertC = 8;
	baseLevel.faces[1].vertA = 5;	baseLevel.faces[1].vertB = 8;	baseLevel.faces[1].vertC = 10;
	baseLevel.faces[2].vertA = 6;	baseLevel.faces[2].vertB = 7;	baseLevel.faces[2].vertC = 12;
	baseLevel.faces[3].vertA = 6;	baseLevel.faces[3].vertB = 11;	baseLevel.faces[3].vertC = 7;
	baseLevel.faces[4].vertA = 1;	baseLevel.faces[4].vertB = 5;	baseLevel.faces[4].vertC = 4;
	baseLevel.faces[5].vertA = 1;	baseLevel.faces[5].vertB = 4;	baseLevel.faces[5].vertC = 6;
	baseLevel.faces[6].vertA = 3;	baseLevel.faces[6].vertB = 8;	baseLevel.faces[6].vertC = 2;
	baseLevel.faces[7].vertA = 3;	baseLevel.faces[7].vertB = 2;	baseLevel.faces[7].vertC = 7;
	baseLevel.faces[8].vertA = 9;	baseLevel.faces[8].vertB = 1;	baseLevel.faces[8].vertC = 12;
	baseLevel.faces[9].vertA = 9;	baseLevel.faces[9].vertB = 12;	baseLevel.faces[9].vertC = 2;
	baseLevel.faces[10].vertA = 10;	baseLevel.faces[10].vertB = 11;	baseLevel.faces[10].vertC = 4;
	baseLevel.faces[11].vertA = 10;	baseLevel.faces[11].vertB = 3;	baseLevel.faces[11].vertC = 11;
	baseLevel.faces[12].vertA = 9;	baseLevel.faces[12].vertB = 5;	baseLevel.faces[12].vertC = 1;
	baseLevel.faces[13].vertA = 12;	baseLevel.faces[13].vertB = 1;	baseLevel.faces[13].vertC = 6;
	baseLevel.faces[14].vertA = 5;	baseLevel.faces[14].vertB = 10;	baseLevel.faces[14].vertC = 4;
	baseLevel.faces[15].vertA = 6;	baseLevel.faces[15].vertB = 4;	baseLevel.faces[15].vertC = 11;
	baseLevel.faces[16].vertA = 8;	baseLevel.faces[16].vertB = 9;	baseLevel.faces[16].vertC = 2;
	baseLevel.faces[17].vertA = 7;	baseLevel.faces[17].vertB = 2;	baseLevel.faces[17].vertC = 12;
	baseLevel.faces[18].vertA = 8;	baseLevel.faces[18].vertB = 3;	baseLevel.faces[18].vertC = 10;
	baseLevel.faces[19].vertA = 7;	baseLevel.faces[19].vertB = 11;	baseLevel.faces[19].vertC = 3;
}

bool SubdivisionSphere::performSubdivision(Level* prevLevel, Level* nextLevel)
{
	std::vector<int> *i = new std::vector<int>();
	std::vector<int> *j = new std::vector<int>();

	nextLevel->levelIndex = prevLevel->levelIndex + 1;
	//


#pragma region initialIandJ

	//initial i
	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertA);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertB);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertC);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertB);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertC);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		i->push_back(prevLevel->faces[iter].vertA);
	}

	//initial j
	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertB);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertC);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertA);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertA);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertB);
	}

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		j->push_back(prevLevel->faces[iter].vertC);
	}

#pragma endregion

	std::vector<int>* I = new std::vector<int>();

	for (int iter = 0; iter < i->size(); iter++)
	{
		if ((*i)[iter] < (*j)[iter])
		{
			I->push_back(iter);
		}
	}

	std::vector<int> *i_tmp = new std::vector<int>();
	std::vector<int> *j_tmp = new std::vector<int>();

	for (int iter = 0; iter < I->size(); iter++)
	{
		i_tmp->push_back((*i)[(*I)[iter]]);
		j_tmp->push_back((*j)[(*I)[iter]]);
	}
	delete i;
	delete j;
	i = i_tmp;
	j = j_tmp;

	//std::vector<int> tmpC;
	//for (size_t iter = 0; iter < i->size(); iter++)
	//{
	//	tmpC.push_back((*i)[iter] + 1234567 * (*j)[iter]);
	//}

	std::vector<int>* index= new std::vector<int>((*i).size(), 0);

	for (int i = 0; i != index->size(); i++)
	{
		(*index)[i] = i;
	}

	//std::sort(index.begin(), index.end(),
	//	[&](const int& a, const int& b)
	//			{return (tmpC[a] < tmpC[b]);}
	//	);

	std::sort((*index).begin(), (*index).end(),
		[&](const int& a, const int& b)
	{
		if ((*j)[a] == (*j)[b])
		{
			return ((*i)[a] < (*i)[b]);
		}
		else
		{
			return ((*j)[a] < (*j)[b]);
		}
	}
	);


	

	/*std::vector<int> test;
	for (int iter = 0; iter < index.size(); iter++)
	{
		test.push_back(tmpC[index[iter]]);
	}*/

	I->clear();

	for (int iter = 0; iter < (*index).size() / 2; iter++)
	{
		I->push_back(std::max((*index)[iter * 2 + 1], (*index)[iter * 2]));
	}

	i_tmp = new std::vector<int>();
	j_tmp = new std::vector<int>();

	for (int iter = 0; iter < I->size(); iter++)
	{
		i_tmp->push_back((*i)[(*I)[iter]]);
		j_tmp->push_back((*j)[(*I)[iter]]);
	}
	delete i; delete j; delete I;


	i = i_tmp;
	j = j_tmp;

	std::vector<int>* s = new std::vector<int>();
	for (int iter = 0; iter < i->size(); iter++)
	{

		s->push_back(prevLevel->numberOfVertices + 1 + iter);
	}

	std::map <std::pair<int, int>, int>* A = new std::map<std::pair<int, int>, int>();


	/*if (prevLevel->levelIndex == 7)
	{
		std::cout << "stop" << std::endl;
	}*/

	for (int iter = 0; iter < s->size(); iter++)
	{
		(*A)[std::make_pair((*i)[iter], (*j)[iter])] = (*s)[iter];
		(*A)[std::make_pair((*j)[iter], (*i)[iter])] = (*s)[iter];
	}


	std::vector<int>* v12 = new std::vector<int>();
	std::vector<int>* v23 = new std::vector<int>();
	std::vector<unsigned long long int>* v31 = new std::vector<unsigned long long int>();

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		unsigned long long int k, l, num;
		unsigned long long int test = (unsigned long long int)((*prevLevel).faces[iter].vertB - 1)*prevLevel->numberOfVertices;
		num = (*prevLevel).faces[iter].vertA + (unsigned long long int)((*prevLevel).faces[iter].vertB - 1)*prevLevel->numberOfVertices;
		k = (num - 1) / prevLevel->numberOfVertices + 1;
		l = (num - 1) % prevLevel->numberOfVertices + 1;
		v12->push_back((*A)[std::make_pair(k, l)]);

		num = (*prevLevel).faces[iter].vertB + (unsigned long long int)((*prevLevel).faces[iter].vertC - 1)*prevLevel->numberOfVertices;
		k = (num - 1) / prevLevel->numberOfVertices + 1;
		l = (num - 1) % prevLevel->numberOfVertices + 1;
		v23->push_back((*A)[std::make_pair(k, l)]);

		num = (*prevLevel).faces[iter].vertC + (unsigned long long int)((*prevLevel).faces[iter].vertA - 1)*prevLevel->numberOfVertices;
		k = (num - 1) / prevLevel->numberOfVertices + 1;
		l = (num - 1) % prevLevel->numberOfVertices + 1;
		v31->push_back((*A)[std::make_pair(k, l)]);
	}

	nextLevel->faces = new Face[prevLevel->numberOfFaces * 4];
	nextLevel->numberOfFaces = prevLevel->numberOfFaces * 4;

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{
		nextLevel->faces[iter].vertA = prevLevel->faces[iter].vertA;
		nextLevel->faces[iter].vertB = (*v12)[iter];
		nextLevel->faces[iter].vertC = (*v31)[iter];
		prevLevel->faces[iter].childFaceA = iter;
		nextLevel->faces[iter].parentFace = iter;

		nextLevel->faces[iter + prevLevel->numberOfFaces].vertA = prevLevel->faces[iter].vertB;
		nextLevel->faces[iter + prevLevel->numberOfFaces].vertB = (*v23)[iter];
		nextLevel->faces[iter + prevLevel->numberOfFaces].vertC = (*v12)[iter];
		prevLevel->faces[iter].childFaceB = prevLevel->numberOfFaces + iter;
		nextLevel->faces[iter + prevLevel->numberOfFaces].parentFace = iter;

		nextLevel->faces[iter + 2 * prevLevel->numberOfFaces].vertA = prevLevel->faces[iter].vertC;
		nextLevel->faces[iter + 2 * prevLevel->numberOfFaces].vertB = (*v31)[iter];
		nextLevel->faces[iter + 2 * prevLevel->numberOfFaces].vertC = (*v23)[iter];
		prevLevel->faces[iter].childFaceC = 2 * prevLevel->numberOfFaces + iter;
		nextLevel->faces[iter + 2 * prevLevel->numberOfFaces].parentFace = iter;


		nextLevel->faces[iter + 3 * prevLevel->numberOfFaces].vertA = (*v12)[iter];
		nextLevel->faces[iter + 3 * prevLevel->numberOfFaces].vertB = (*v23)[iter];
		nextLevel->faces[iter + 3 * prevLevel->numberOfFaces].vertC = (*v31)[iter];
		prevLevel->faces[iter].childFaceD = 3 * prevLevel->numberOfFaces + iter;
		nextLevel->faces[iter + 3 * prevLevel->numberOfFaces].parentFace = iter;


	}

	nextLevel->vertices = new Vertex[prevLevel->numberOfVertices + i->size()];
	nextLevel->numberOfVertices = prevLevel->numberOfVertices + (int)i->size();

	for (int iter = 0; iter < prevLevel->numberOfVertices; iter++)
	{
		nextLevel->vertices[iter] = prevLevel->vertices[iter];
	}

	for (size_t iter = 0; iter < i->size(); iter++)
	{
		nextLevel->vertices[iter + prevLevel->numberOfVertices].creationLevel = nextLevel->levelIndex;
		nextLevel->vertices[iter + prevLevel->numberOfVertices].parentA = (*i)[iter];
		nextLevel->vertices[iter + prevLevel->numberOfVertices].parentB = (*j)[iter];

		nextLevel->vertices[iter + prevLevel->numberOfVertices].position = glm::normalize((nextLevel->vertices[(*i)[iter] - 1].position + nextLevel->vertices[(*j)[iter] - 1].position) / 2.0f);

		//nextLevel->vertices[iter + prevLevel->numberOfVertices].position.y = (nextLevel->vertices[(*i)[iter]-1].position.y + nextLevel->vertices[(*j)[iter]-1].position.y)/2.0f;
		//nextLevel->vertices[iter + prevLevel->numberOfVertices].position.z = (nextLevel->vertices[(*i)[iter]-1].z + nextLevel->vertices[(*j)[iter]-1].z)/2.0f;

		//float d = sqrt(pow(nextLevel->vertices[iter + prevLevel->numberOfVertices].x, 2) +
		//	pow(nextLevel->vertices[iter + prevLevel->numberOfVertices].y, 2) +
		//	pow(nextLevel->vertices[iter + prevLevel->numberOfVertices].z, 2));



		//nextLevel->vertices[iter + prevLevel->numberOfVertices].x = nextLevel->vertices[iter + prevLevel->numberOfVertices].x /= d;
		//nextLevel->vertices[iter + prevLevel->numberOfVertices].y = nextLevel->vertices[iter + prevLevel->numberOfVertices].y /= d;
		//nextLevel->vertices[iter + prevLevel->numberOfVertices].z = nextLevel->vertices[iter + prevLevel->numberOfVertices].z /= d;
	}

	delete i;
	delete j;	
	delete A;
	delete s;
	delete v12; delete v23; delete v31;
	return true;
}


/*int SubdivisionShpere::SubdivisionSphere::vectorToIndex(const glm::vec3& vect, int level, std::vector<std::vector<Face>> testRes)
{

	double minDist = std::numeric_limits<double>::max();
	int indexOfItitFace = -1;
	double dist = std::numeric_limits<double>::max();
	for (int iter = 0; iter < levels[0].numberOfFaces; iter++)
	{
		dist = glm::length(levels[0].vertices[levels[0].faces[iter].vertA].position - vect)+
			glm::length(levels[0].vertices[levels[0].faces[iter].vertB].position - vect)+
			glm::length(levels[0].vertices[levels[0].faces[iter].vertC].position - vect);


		if (dist < minDist)
		{
			minDist = dist;
			indexOfItitFace = iter;
		}
	}
	Face* face = &(levels[0].faces[indexOfItitFace]);
	double distA, distB, distC, distD;
	for (int iter = 1; iter <= level; iter++)
	{
		distA = glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceA].vertA].position - vect) +
				  glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceA].vertB].position - vect) +
				  glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceA].vertC].position - vect);

		distB = glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceB].vertA].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceB].vertB].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceB].vertC].position - vect);

		distC = glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceC].vertA].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceC].vertB].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceC].vertC].position - vect);

		distD = glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceD].vertA].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceD].vertB].position - vect) +
			glm::length(levels[iter].vertices[levels[iter].faces[face->childFaceD].vertC].position - vect);


		minDist = distA;
		Face* nextFace = NULL;
		nextFace = &(levels[iter].faces[face->childFaceA]);

		if (distB < minDist)
		{
			minDist = distB;
			nextFace = &(levels[iter].faces[face->childFaceB]);
		}

		if (distC < minDist)
		{
			minDist = distC;
			nextFace = &(levels[iter].faces[face->childFaceC]);
		}

		if (distD < minDist)
		{
			minDist = distD;
			nextFace = &(levels[iter].faces[face->childFaceD]);
		}
		face = nextFace;
	}


	distA = glm::length(levels[level].vertices[face->vertA].position - vect);
	distB = glm::length(levels[level].vertices[face->vertB].position - vect);
	distC = glm::length(levels[level].vertices[face->vertC].position - vect);

	minDist = distA;
	int index = face->vertA;
	if (distB < minDist)
	{
		minDist = distB;
		index = face->vertB;
	}
	if (distC < minDist)
	{
		minDist = distC;
		index = face->vertC;
	}
	return index;
}*/

int SubdivisionShpere::SubdivisionSphere::vectorToFaceIndex(const glm::vec3& vect, int level)
{

	Face* face = NULL; 
	Face* newFace = NULL;
	for (int iter = 0; iter < levels[0].numberOfFaces; iter++)
	{
		
		if (testTriangle(	levels[0].faces[iter].vertARef->position,
							levels[0].faces[iter].vertBRef->position,
							levels[0].faces[iter].vertCRef->position,
			vect))
		{
			face = &(levels[0].faces[iter]);
			break;
		}		
	}


	for (int iter = 1; iter <= level; iter++)
	{
		if (sameSide(face->childFaceARef->vertBRef->position, face->childFaceARef->vertCRef->position, face->childFaceARef->vertARef->position, vect))
		{
			face = face->childFaceARef;
			continue;
		}
		if (sameSide(face->childFaceBRef->vertBRef->position, face->childFaceBRef->vertCRef->position, face->childFaceBRef->vertARef->position, vect))
		{
			face = face->childFaceBRef;
			continue;
		}
		if (sameSide(face->childFaceCRef->vertBRef->position, face->childFaceCRef->vertCRef->position, face->childFaceCRef->vertARef->position, vect))
		{
			face = face->childFaceCRef;
			continue;
		}
		face = face->childFaceDRef;

		/*if (testTriangle(	face->childFaceARef->vertARef->position,
							face->childFaceARef->vertBRef->position,
							face->childFaceARef->vertCRef->position,
			vect))
		{
			face = face->childFaceARef;
			continue;
		}
		if (testTriangle(	face->childFaceBRef->vertARef->position,
							face->childFaceBRef->vertBRef->position,
							face->childFaceBRef->vertCRef->position,
			vect))
		{
			face = face->childFaceBRef;
			continue;
		}
		if (testTriangle(	face->childFaceCRef->vertARef->position,
							face->childFaceCRef->vertBRef->position,
							face->childFaceCRef->vertCRef->position,
			vect))
		{
			face = face->childFaceCRef;
			continue;
		}
		if (testTriangle(	face->childFaceDRef->vertARef->position,
							face->childFaceDRef->vertBRef->position,
							face->childFaceDRef->vertCRef->position,
			vect))
		{
			face = face->childFaceDRef;
			continue;
		}*/			
	}

	return face->index;
}





