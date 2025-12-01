from build_tools.directory import Directory
from build_tools.scan import Scanner
from build_tools.linker import Linker
from build_tools.cpp_compile import CppCompiler
from build_tools.cu_compile import CuCompiler


def main():
	cpp_files, cu_files = Scanner.getStaleFiles()

	CppCompiler.compileListST(cpp_files)
	CuCompiler.compileListST(cu_files)
	

	obj_files = Scanner.getObjectFiles()
	Linker.link(obj_files)


	

if __name__ == "__main__":
	main() #run main on run