#include "stdafx.h"
#include "Parameters.h"
#include <string>

Parameters::Parameters(int argc, char** argv)
{
	//shownLevel = numberOfLevels;
	for (int i = 1; i < argc; i++)
	{
		try
		{
			std::string nameOfParam = argv[i];
			if (nameOfParam == "-NumOfLevels")
			{
				try
				{
					numberOfLevels = std::stoi((std::string)(argv[i + 1]));
				}
				catch (const std::exception& )
				{
					std::cout << "-NumOfLevels n -PathToRWR path" << std::endl;
				}
			}
			if (nameOfParam == "-ScaleLuminance")
			{
				try
				{
					scaleLuminance = std::stod((std::string)argv[i + 1]);
				}
				catch (const std::exception&)
				{
					std::cout << "-NumOfLevels n -PathToRWR path -ScaleLuminance scale" << std::endl;
				}

			}

			if (nameOfParam == "-VR")
			{
				vrMode = true;
			}

			if (nameOfParam == "-PathToRWR")
			{
				try
				{
					pathToRayFile = (std::string)(argv[i + 1]);
					startMode = StartMode::RayFile;

				}
				catch (const std::exception&)
				{
					std::cout << "-NumOfLevels n -PathToRWR path" << std::endl;
				}
			}

		}
		catch (const std::exception&)
		{
			std::cout << "-NumOfLevels n -PathToRWR path -ScaleLuminance scale" << std::endl;
		}

	}
}
