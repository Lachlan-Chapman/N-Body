import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from build_tools.directory import Directory

class CppCompiler:
	CXX: str = "g++"
	THREAD_COUNT: int = 0  #0 = use all system threads else use this number for threads

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
		"-I/usr/local/cuda/include",
	]

	LIBS: list[str] = []

	@classmethod
	def compileListST(cls, files: list[tuple[Path, Path]]) -> bool:
		for src, obj in files:
			print(f"-- Compiling C++ (ST): {src} -> {obj} --")
			if not cls._compileFile(src, obj):
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

				#result() returns True/False instead of throwing to stop vscode thinking the program has failed (failing subprocess is okay)
				if not fut.result():
					print(f"!~ Compilation Failed: {src} ~!")

					#cancel all other tasks if one of them fails
					for f in futures:
						f.cancel()

					return False

		return True

	@classmethod
	def _compileFile(cls, src: Path, obj: Path) -> bool:
		cmd = [
			cls.CXX,
			"-c", str(src),
			"-o", str(obj),
			*cls.FLAGS,
			*cls.INCLUDE,
			*cls.LIBS,
		]

		proc = subprocess.run(cmd, capture_output=True, text=True)

		if proc.returncode != 0:
			print(proc.stdout)
			print(proc.stderr)
			return False

		return True
