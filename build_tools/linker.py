import subprocess
from pathlib import Path
from build_tools.directory import Directory

class Linker:
	CXX: str = "g++"
	LIB_DIRS: list[str] = [
		"-L/usr/local/cuda/lib64"
	]

	LIBS: list[str] = [
		"-lcudart", "-lglfw", "-lGL", #cuda, glfw and OpenGL
		"-lX11", "-lXrandr", "-lXi", "-lXxf86vm", "-lXcursor", #needed for linux x11 to link glfw
		"-ldl", "-lpthread" #dyamic loaders
	]
	FLAGS: list[str] = []
	
	@classmethod
	def link(cls, object_files: list[Path]) -> bool:
		if not object_files:
			print("++ No object files — skipping link ++")
			return True

		print(f"** Linking -> [{Directory.BUILD_DIRECTORY / Directory.PROGRAM_NAME}] **")

		cmd = [
			cls.CXX,
			*map(str, object_files),
			"-o", str(Directory.BUILD_DIRECTORY / Directory.PROGRAM_NAME),
			*cls.FLAGS,
			*cls.LIB_DIRS,
			*cls.LIBS,
		]

		link_process = subprocess.run(cmd)
		if link_process.returncode != 0:
			print("!! Linking Failed !!")
			print(link_process.stdout)
			print(link_process.stderr)
			return False