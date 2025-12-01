#holds all info relating to where files can be found etc

from pathlib import Path

class Directory:
	SRC_DIRECTORY = Path("src")
	INCLUDE_DIRECTORY = Path("include")
	OBJECT_DIRECTORY = Path("object")
	BUILD_DIRECTORY = Path("build")
	PROGRAM_NAME = "app"

	@classmethod
	def verifyFolders(cls) -> bool:
		required = [
			cls.SRC_DIRECTORY,
			cls.INCLUDE_DIRECTORY,
			cls.OBJECT_DIRECTORY,
			cls.BUILD_DIRECTORY,
		]

		for folder in required:
			try:
				#Make directory if missing
				folder.mkdir(exist_ok=True)
			except Exception as e:
				print(f"!! Failed to create directory: {folder} !!")
				print(e)
				return False

		return True

	@classmethod
	def removeBuild(cls) -> bool:
		exe = cls.BUILD_DIRECTORY / cls.PROGRAM_NAME
		if exe.exists():
			exe.unlink()
		return True