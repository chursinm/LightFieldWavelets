#include "SubdivisionSphere.h"



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
		performSubdivision( &(levels[iter - 1]), &(levels[iter]));
	}

}

void SubdivisionSphere::initLevel()
{
	Level& baseLevel = levels[0];
	baseLevel.vertices = new Vertex[12];
	baseLevel.numberOfVertices = 12;
	baseLevel.faces = new Face[20];
	baseLevel.numberOfFaces = 20;
	float tau=  0.8506508084;
	float one = 0.5257311121;
	//vertices 
	baseLevel.vertices[ 0].x =  tau;	baseLevel.vertices[ 0].y =  one;    baseLevel.vertices[ 0].z =    0;
	baseLevel.vertices[ 1].x = -tau;	baseLevel.vertices[ 1].y =  one;	baseLevel.vertices[ 1].z =    0;
	baseLevel.vertices[ 2].x = -tau;	baseLevel.vertices[ 2].y = -one;    baseLevel.vertices[ 2].z =    0;
	baseLevel.vertices[ 3].x =  tau;	baseLevel.vertices[ 3].y = -one;    baseLevel.vertices[ 3].z =    0;
	baseLevel.vertices[ 4].x =  one;	baseLevel.vertices[ 4].y =    0;    baseLevel.vertices[ 4].z =  tau;
	baseLevel.vertices[5].x =   one; 	baseLevel.vertices[ 5].y =    0;    baseLevel.vertices[ 5].z = -tau;
	baseLevel.vertices[ 6].x = -one;	baseLevel.vertices[ 6].y =    0;    baseLevel.vertices[ 6].z = -tau;
	baseLevel.vertices[ 7].x = -one;	baseLevel.vertices[ 7].y =    0;    baseLevel.vertices[ 7].z =  tau;
	baseLevel.vertices[ 8].x =    0;	baseLevel.vertices[ 8].y =  tau;    baseLevel.vertices[ 8].z =  one;
	baseLevel.vertices[ 9].x =    0;	baseLevel.vertices[ 9].y = -tau;    baseLevel.vertices[ 9].z =  one;
	baseLevel.vertices[10].x =    0;	baseLevel.vertices[10].y = -tau;    baseLevel.vertices[10].z = -one;
	baseLevel.vertices[11].x =    0;	baseLevel.vertices[11].y =  tau;    baseLevel.vertices[11].z = -one;
	//faces

	baseLevel.faces[ 0].vertA =  5;	baseLevel.faces[ 0].vertB =  9;	baseLevel.faces[ 0].vertC =  8;
	baseLevel.faces[ 1].vertA =  5;	baseLevel.faces[ 1].vertB =  8;	baseLevel.faces[ 1].vertC = 10;
	baseLevel.faces[ 2].vertA =  6;	baseLevel.faces[ 2].vertB =  7;	baseLevel.faces[ 2].vertC = 12;
	baseLevel.faces[ 3].vertA =  6;	baseLevel.faces[ 3].vertB = 11;	baseLevel.faces[ 3].vertC =  7;
	baseLevel.faces[ 4].vertA =  1;	baseLevel.faces[ 4].vertB =  5;	baseLevel.faces[ 4].vertC =  4;
	baseLevel.faces[ 5].vertA =  1;	baseLevel.faces[ 5].vertB =  4;	baseLevel.faces[ 5].vertC =  6;
	baseLevel.faces[ 6].vertA =  3;	baseLevel.faces[ 6].vertB =  8;	baseLevel.faces[ 6].vertC =  2;
	baseLevel.faces[ 7].vertA =  3;	baseLevel.faces[ 7].vertB =  2;	baseLevel.faces[ 7].vertC =  7;
	baseLevel.faces[ 8].vertA =  9;	baseLevel.faces[ 8].vertB =  1;	baseLevel.faces[ 8].vertC = 12;
	baseLevel.faces[ 9].vertA =  9;	baseLevel.faces[ 9].vertB = 12;	baseLevel.faces[ 9].vertC =  2;
	baseLevel.faces[10].vertA = 10;	baseLevel.faces[10].vertB = 11;	baseLevel.faces[10].vertC =  4;
	baseLevel.faces[11].vertA = 10;	baseLevel.faces[11].vertB =  3;	baseLevel.faces[11].vertC = 11;
	baseLevel.faces[12].vertA =  9;	baseLevel.faces[12].vertB =  5;	baseLevel.faces[12].vertC =  1;
	baseLevel.faces[13].vertA = 12;	baseLevel.faces[13].vertB =  1;	baseLevel.faces[13].vertC =  6;
	baseLevel.faces[14].vertA =  5;	baseLevel.faces[14].vertB = 10;	baseLevel.faces[14].vertC =  4;
	baseLevel.faces[15].vertA =  6;	baseLevel.faces[15].vertB =  4;	baseLevel.faces[15].vertC = 11;
	baseLevel.faces[16].vertA =  8;	baseLevel.faces[16].vertB =  9;	baseLevel.faces[16].vertC =  2;
	baseLevel.faces[17].vertA =  7;	baseLevel.faces[17].vertB =  2;	baseLevel.faces[17].vertC = 12;
	baseLevel.faces[18].vertA =  8;	baseLevel.faces[18].vertB =  3;	baseLevel.faces[18].vertC = 10;
	baseLevel.faces[19].vertA =  7;	baseLevel.faces[19].vertB = 11;	baseLevel.faces[19].vertC =  3;	
}

bool SubdivisionSphere::performSubdivision(Level* prevLevel, Level* nextLevel)
{
	std::vector<int> *i = new std::vector<int>();
	std::vector<int> *j = new std::vector<int>();



	std::vector<int> *I = new std::vector<int>();


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

	std::vector<int> tmpC;
	for (int iter = 0; iter < i->size(); iter++)
	{
		tmpC.push_back((*i)[iter] + 1234567 * (*j)[iter]);
	}

	std::vector<int> index(tmpC.size(), 0);

	for (int i = 0; i != index.size(); i++)
	{
		index[i] = i;
	}

	std::sort(index.begin(), index.end(),
		[&](const int& a, const int& b)
				{return (tmpC[a] < tmpC[b]);}
		);

	
	I->clear();

	std::vector<int> test;
	for (int iter = 0; iter < index.size(); iter++)
	{
		test.push_back(tmpC[index[iter]]);
	}

	
	for (int iter = 0; iter < index.size()/2; iter++)
	{		
		I->push_back(std::max(index[iter*2+1], index[iter * 2 ]));
	}

	i_tmp = new std::vector<int>();
	j_tmp = new std::vector<int>();

	for (int iter = 0; iter < I->size(); iter++)
	{
		i_tmp->push_back((*i)[(*I)[iter]]);
		j_tmp->push_back((*j)[(*I)[iter]]);
	}
	delete i; delete j;

	i = i_tmp;
	j = j_tmp;
	
	std::vector<int> s;
	for (int iter = 0; iter < i->size(); iter++)
	{
		s.push_back(prevLevel->numberOfVertices + 1 + iter);
	}

	std::map <std::pair<int, int>, int> A;

	for (int iter = 0; iter < s.size(); iter++)
	{
		A[std::make_pair((*i)[iter], (*j)[iter])] = s[iter];
		A[std::make_pair((*j)[iter], (*i)[iter])] = s[iter];
	}

	std::vector<int> v12, v23, v31;

	for (int iter = 0; iter < prevLevel->numberOfFaces; iter++)
	{	
		if (iter == 13)
		{
			std::cout << "stop" << std::endl;
		}
		int k, l, num;
		num = (*prevLevel).faces[iter].vertA + ((*prevLevel).faces[iter].vertB - 1)*prevLevel->numberOfVertices;		
		k = (num-1)/prevLevel->numberOfVertices +1;
		l = (num-1) % prevLevel->numberOfVertices +1;
		v12.push_back(A[std::make_pair(k,l)]);

		num = (*prevLevel).faces[iter].vertB + ((*prevLevel).faces[iter].vertC - 1)*prevLevel->numberOfVertices;
		k = (num - 1) / prevLevel->numberOfVertices + 1;
		l = (num - 1) % prevLevel->numberOfVertices + 1;
		v23.push_back(A[std::make_pair(k, l)]);

		num = (*prevLevel).faces[iter].vertC + ((*prevLevel).faces[iter].vertA - 1)*prevLevel->numberOfVertices;
		k = (num - 1) / prevLevel->numberOfVertices + 1;
		l = (num - 1) % prevLevel->numberOfVertices + 1;
		v31.push_back(A[std::make_pair(k, l)]);

	}

	

	return true;
}




