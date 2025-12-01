import subprocess
from pathlib import Path
from build_tools.directory import Directory

class Linker:
	CXX: str = "g++"
	LIB_DIRS: list[str] = [
		"-L/usr/local/cuda/lib64"
	]

	LIBS: list[str] = [
		"-lcudart"
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

		link_process = subprocess.run(cmd, check=True)
		if link_process.returncode != 0:
			print("!! Linking Failed !!")
			print(proc.stdout)
			print(proc.stderr)
			return false
		return True
