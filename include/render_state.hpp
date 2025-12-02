#pragma once
enum class rendererState : int {
	INITIALIZING = 0,
	LOADING = 1,
	RUNNING = 2,
	SHUTTING_DOWN = 3,
	IDLE = 4
};