#pragma once

enum class StartMode
{
	RayFile,
	Checker
};

class Parameters
{
public:
	StartMode startMode = StartMode::Checker;
	int numberOfLevels = 4;
	double yCoord = 1.3f;
	double zCoord = -1.5f;
	std::string pathToRayFile = "";
	double scaleLuminance=2.5;
	bool vrMode = false;
	Parameters(int argc, char** argv);
};