import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from build_tools.directory import Directory

class CppCompiler:
	CXX: str = "g++"
	THREAD_COUNT: int = 0  #0 = use all system threads

	FLAGS: list[str] = [
		"-std=c++20",
		"-MMD",
		"-MP",
		"-g",
		"-O0",
		"-Wall",
		"-Wextra",
	]

	INCLUDE: list[str] = [
		f"-I{Directory.INCLUDE_DIRECTORY}",
	]

	LIBS: list[str] = []  #good for later use if we start to use openGL

	@classmethod
	def compileListST(cls, files: list[tuple[Path, Path]]) -> bool:
		for src, obj in files:
			print(f"-- Compiling C++ (ST): {src} -> {obj} --")

			try:
				cls._compileFile(src, obj)
			except subprocess.CalledProcessError:
				print(f"!~ Compilation Failed: {src} ~!")
				return False

		return True

	@classmethod
	def compileListMT(cls, files: list[tuple[Path, Path]]) -> bool:
		thread_count: int = cls.THREAD_COUNT or (os.cpu_count() or 1)

		print(f"-- Compiling C++ (MT | {thread_count} threads) --")

		with ThreadPoolExecutor(max_workers=thread_count) as pool:
			futures = {
				pool.submit(cls._compileFile, src, obj): (src, obj)
				for src, obj in files
			}

			for fut in as_completed(futures):
				src, obj = futures[fut]

				try:
					fut.result()
				except subprocess.CalledProcessError:
					print(f"!~ Compilation Failed: {src} ~!")

					for f in futures:
						f.cancel()

					return False

		return True

	@classmethod
	def _compileFile(cls, src: Path, obj: Path) -> None:
		cmd = [
			cls.CXX,
			"-c", str(src),
			"-o", str(obj),
			*cls.FLAGS,
			*cls.INCLUDE,
			*cls.LIBS,
		]

		subprocess.run(cmd, check=True)
