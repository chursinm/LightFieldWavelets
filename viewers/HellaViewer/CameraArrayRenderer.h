#pragma once
#include "CameraArrayParser.h"
class CameraArrayRenderer
{
public:
	CameraArrayRenderer();
	~CameraArrayRenderer();
	void initialize();
	void update();
	void render();
private:
	CameraArray m_CameraArray;
};

