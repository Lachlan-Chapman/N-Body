import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from build_tools.directory import Directory

class CuCompiler:
	CXX: str = "nvcc"
	LIBS: list[str] = []
	FLAGS: list[str] = [
		"--std",
		"c++20",
		"-arch=sm_89"
	]
	HOST_FLAGS: list[str] = [
		"--std=c++20"
	]
	INCLUDES: list[str] = [f"-I{Directory.INCLUDE_DIRECTORY}"]
	THREAD_COUNT: int = 0 #0 means use max threads of the system

	@classmethod
	def compileListST(cls, files: list[tuple[Path, Path]]) -> bool:
		for src, obj in files:
			print(f"-- Compiling Cuda C (ST): {src} -> {obj} --")

			try:
				cls._compileFile(src, obj)
			except subprocess.CalledProcessError:
				print(f"!~ Compilation Failed: {src} ~!")
				return False

		return True

	@classmethod
	def compileListMT(cls, files: list[tuple[Path, Path]]) -> bool:
		thread_count: int = os.cpu_count() or 1

		if cls.THREAD_COUNT != 0:
			thread_count = cls.THREAD_COUNT

		print(f"-- Compiling Cuda C (MT | {thread_count} threads) --")

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

					# cancel all other futures
					for f in futures:
						f.cancel()

					return False

		return True

	@classmethod
	def _compileFile(cls, src: Path, obj: Path) -> None:
		cmd: list[str] = [
			cls.CXX,
			"-c", str(src),
			"-o", str(obj),
			*cls.FLAGS,
			*cls.HOST_FLAGS,
			*cls.LIBS,
		]
		subprocess.run(cmd, check=True)
