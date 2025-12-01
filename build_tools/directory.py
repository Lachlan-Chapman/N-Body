#holds all info relating to where files can be found etc

from pathlib import Path

class Directory:
	SRC_DIRECTORY = Path("src")
	INCLUDE_DIRECTORY = Path("include")
	OBJECT_DIRECTORY = Path("object")
	BUILD_DIRECTORY = Path("build")

	@classmethod
	def verifyFolders() -> bool:
		return True
