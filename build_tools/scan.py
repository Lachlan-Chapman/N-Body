from pathlib import Path

from build_tools.directory import Directory

class Scanner:
	@classmethod
	def getObjectFiles(cls) -> list[Path]:
		return list(Directory.OBJECT_DIRECTORY.glob("*.o")) #glob  takes a pattern arg

	@classmethod
	def getStaleFiles(cls) -> tuple[list[tuple[Path, Path], list[tuple[Path, Path]]]]:
		stale_cpp: list[tuple[Path, Path]] = []
		stale_cu: list[tuple[Path, Path]] = []

		stale_cpp.extend(cls._staleCpp()) #.cpp files without an .o file OR .o is older than the .cpp file
		stale_cpp.extend(cls._modifiedDependenciesCpp()) #.cpp files with newer .hpp files than the .o file
		
		unique: dict[Path, tuple[Path, Path]] = {}
		for src, obj in stale_cpp:
			unique[src] = (src, obj) #use dictionary to store only 1 instance of (src, obj) pair
		stale_cpp[:] = list(unique.values()) #allow inplace overwrite


		stale_cu.extend(cls._staleCu())
		stale_cu.extend(cls._modifiedDependenciesCu())

		return stale_cpp, stale_cu

	#gets the cpp files needed for compilation based on IF .o is missing or .cpp is modified
	@classmethod
	def _staleCpp(cls) -> list[tuple[Path, Path]]:
		stale: list[tuple[Path, Path]] = []

		for src in Directory.SRC_DIRECTORY.rglob("*.cpp"): #r in rglob means recursive so itll search subfolders
			obj = Directory.OBJECT_DIRECTORY / (src.stem + ".o")

			if not obj.exists(): #no respective object file clearly needs compiling
				stale.append((src, obj))
				continue

			if src.stat().st_mtime > obj.stat().st_mtime: #cpp has updated later than object file was created
				stale.append((src, obj))
				continue

		return stale

	@classmethod
	def _modifiedDependenciesCpp(cls) -> list[tuple[Path, Path]]:
		stale: list[tuple[Path, Path]] = []
		for src in Directory.SRC_DIRECTORY.rglob("*.cpp"):
			obj = Directory.OBJECT_DIRECTORY / (src.stem + ".o")
			dep = Directory.OBJECT_DIRECTORY / (src.stem + ".d")

			#if the dependecy file doesnt exist, its the responsibility elsewhere to find and create them
			if not obj.exists() or not dep.exists(): #no respective object file clearly needs compiling
				continue

			dependencies: list[Path] = cls._parseDependencyFiles(dep) #all the depended on files for this o file
			obj_mtime: float = obj.stat().st_mtime

			for header_path in dependencies:
				if header_path.exists() and header_path.stat().st_mtime > obj_mtime: #re compile files with changes to the included headers
					stale.append((src, obj))
					break
		return stale

	@classmethod
	def _parseDependencyFiles(cls, dep_path: Path) -> list[Path]:
		try:
			text = dep_path.read_text().replace("\\", "")
		except:
			return [] #the file given wasnt a d file OR it was empty
		
		parts = text.split(":") #parts[1] has the depended on files
		if(len(parts) < 2):
			return [] #this would mean no depended on files are listed
		dependencies = parts[1].strip().split()
		return [Path(d) for d in dependencies]

	@classmethod
	def _staleCu(cls) -> list[tuple[Path, Path]]:
		stale: list[tuple[Path, Path]] = []

		for src in Directory.SRC_DIRECTORY.rglob("*.cu"): #r in rglob means recursive so itll search subfolders
			obj = Directory.OBJECT_DIRECTORY / (src.stem + ".o")

			if not obj.exists(): #no respective object file clearly needs compiling
				stale.append((src, obj))
				continue

			if src.stat().st_mtime > obj.stat().st_mtime: #cpp has updated later than object file was created
				stale.append((src, obj))
				continue

		return stale

	@classmethod
	def _modifiedDependenciesCu(cls) -> list[tuple[Path, Path]]:
		stale: list[tuple[Path, Path]] = []
		for src in Directory.SRC_DIRECTORY.rglob("*.cu"):
			obj = Directory.OBJECT_DIRECTORY / (src.stem + ".o")
			dep = Directory.OBJECT_DIRECTORY / (src.stem + ".d")

			#if the dependecy file doesnt exist, its the responsibility elsewhere to find and create them
			if not obj.exists() or not dep.exists(): #no respective object file clearly needs compiling
				continue

			dependencies: list[Path] = cls._parseDependencyFiles(dep) #all the depended on files for this o file
			obj_mtime: float = obj.stat().st_mtime

			for header_path in dependencies:
				if header_path.exists() and header_path.stat().st_mtime > obj_mtime: #re compile files with changes to the included headers
					stale.append((src, obj))
					break
		return stale