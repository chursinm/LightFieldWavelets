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
	std::string pathToRayFile = "";
	double scaleLuminance=2.5;
	bool vrMode = false;
	Parameters(int argc, char** argv);
};