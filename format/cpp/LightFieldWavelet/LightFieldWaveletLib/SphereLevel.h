#pragma once
#include <vector>
#include "SphereVertex.h"

namespace LightField
{
	class SphereLevel
	{
	public:
		SphereLevel(SphereLevel* prevLever);				
		SphereLevel();

				int				getIndex()				const			{ return index; }	

		const	SphereVertex&	getVertex(int index)	const			{ return vertices[index]; }
		const	SphereEdge&		getEdge(int index)		const			{ return edges[index]; }
		const	SphereFace&		getFace(int index)		const			{ return faces[index]; }

		const	auto&			getFaces()				const			{ return faces; }
		const	auto&			getEdges()				const			{ return edges; }
		const	auto&			getVertices()			const			{ return vertices; }
	
		
		size_t	getNumberOfVertices()	const			{ return vertices.size();	}
		size_t	getNumberOfEdges()		const			{ return edges.size();		}
		size_t	getNumberOfFaces()		const 			{ return faces.size();		}

	protected:
		std::vector <SphereVertex>	vertices;
		std::vector <SphereEdge>	edges;
		std::vector <SphereFace>	faces;
		int index = 0;
		friend class SubdivisionSphere;
		void initilizeEdges();
	};
}