#include <iostream>
#include <cstdio>

extern "C" void runTestKernel();

int main(int argc, char** argv) {
	std::cout << "Run kernel" << std::endl;
	runTestKernel();
	std::cout << "Done" << std::endl;
	return 0;
}