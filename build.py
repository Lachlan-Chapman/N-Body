import sys

from build_tools.directory import Directory
from build_tools.scan import Scanner
from build_tools.linker import Linker
from build_tools.cpp_compile import CppCompiler
from build_tools.cu_compile import CuCompiler


def main():
	Directory.verifyFolders()

	cpp_files, cu_files = Scanner.getStaleFiles()
	if not cpp_files and not cu_files:
		print("Nothing To Compile")

	if not Directory.removeBuild():
		print("Failed To Remove Build")
		return False

	print(f"Compiling CPP: {cpp_files}")
	if not CppCompiler.compileListST(cpp_files):
		return False


	print(f"Compiling CU: {cu_files}")
	if not CuCompiler.compileListST(cu_files):
		return False


	obj_files = Scanner.getObjectFiles()
	if not Linker.link(obj_files):
		return False

	return True


	

if __name__ == "__main__":
	main()