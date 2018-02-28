#include <iostream>
#include "SubdivisionSphere.h"

using namespace SubdivisionShpere;

int main()
{
	std::cout << "light field wavelet format cpp start" << std::endl;	
	SubdivisionSphere* subsphere = new SubdivisionSphere(12);
	delete subsphere;
	return 0;
}