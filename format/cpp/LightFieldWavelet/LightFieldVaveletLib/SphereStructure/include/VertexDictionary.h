#pragma once
#include <unordered_map>

//vertex dictionary 

struct  Key
{
	long firstIndex;
	long secondIndex;
	bool operator==(const Key &other) const
	{
		return (firstIndex == other.firstIndex
			&& secondIndex == other.secondIndex);
	}
};

struct KeyHasher
{
	long operator()(const Key& k) const
	{
		//return k.firstIndex+k.secondIndex;
		return k.firstIndex;
	}
};

class VertexDictionary
{
public:
	VertexDictionary();
	~VertexDictionary();

	void testDictionary();
	
private:
	std::unordered_map<Key, float, KeyHasher>* _vertexDictionary;
};

VertexDictionary::VertexDictionary()
{
	_vertexDictionary = new std::unordered_map<Key, float, KeyHasher>();
}

VertexDictionary::~VertexDictionary()
{
	_vertexDictionary;
}

void VertexDictionary::testDictionary()
{
	Key testKey;
	testKey.firstIndex = 123;
	testKey.secondIndex = 321;
	_vertexDictionary->insert(std::make_pair(testKey, 1.0f));

}