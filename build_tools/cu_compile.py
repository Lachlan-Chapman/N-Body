import subprocess
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from build_tools.directory import Directory

def getGitHash():
	try:
		hash = subprocess.run(
			["git", "rev-parse", "--short", "HEAD"],
			capture_output = True,
			text = True,
			check = True
		)
		return hash.stdout.strip()
	except:
		return "idk"

class CuCompiler:
	CXX: str = "nvcc"
	THREAD_COUNT: int = 0

	FLAGS: list[str] = [
		"--std=c++20",
		f'-DGIT_HASH="{getGitHash()}"',
		"-arch=sm_89",
		"--diag-suppress=20012"
	]

	HOST_FLAGS: list[str] = [
		"--std=c++20"
	]

	INCLUDES: list[str] = [
		f"-I{Directory.INCLUDE_DIRECTORY}"
	]

	LIBS: list[str] = []

	@classmethod
	def compileListST(cls, files: list[tuple[Path, Path]]) -> bool:
		for src, obj in files:
			print(f"-- Compiling CUDA (ST): {src} -> {obj} --")
			if not cls._compileFile(src, obj):
				print(f"!~ Compilation Failed: {src} ~!")
				return False
		return True

	@classmethod
	def compileListMT(cls, files: list[tuple[Path, Path]]) -> bool:
		thread_count: int = cls.THREAD_COUNT or (os.cpu_count() or 1)

		print(f"-- Compiling CUDA (MT | {thread_count} threads) --")

		with ThreadPoolExecutor(max_workers=thread_count) as pool:
			futures = {
				pool.submit(cls._compileFile, src, obj): (src, obj)
				for src, obj in files
			}

			for fut in as_completed(futures):
				src, obj = futures[fut]

				if not fut.result():
					print(f"!~ Compilation Failed: {src} ~!")

					# cancel remaining tasks
					for f in futures:
						f.cancel()

					return False

		return True

	@classmethod
	def _compileFile(cls, src: Path, obj: Path) -> bool:
		cmd: list[str] = [
			cls.CXX,
			"-c", str(src),
			"-o", str(obj),
			*cls.FLAGS,
			*cls.HOST_FLAGS,
			*cls.INCLUDES,
			*cls.LIBS,
		]

		proc = subprocess.run(cmd, capture_output=True, text=True)

		if proc.returncode != 0:
			print(proc.stdout)
			print(proc.stderr)
			return False

		return True
