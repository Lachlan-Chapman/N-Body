#pragma once
#include <chrono>
#include <iostream>
class timer {
public:
	virtual ~timer() = default;
	virtual void start() = 0;
	virtual void stop() = 0;
	virtual void reset() = 0;
	virtual void restart() = 0;
	virtual bool running() const = 0;
	virtual double seconds() const = 0;
	virtual double milliseconds() const = 0;
	virtual double microseconds() const = 0;
};

class steadyTimer : public timer {
public:
	steadyTimer();

	void start() override;
	void stop() override;
	void reset() override;
	void restart() override;

	bool running() const override;

	double seconds() const override;
	double milliseconds() const override;
	double microseconds() const override;
protected:
	std::chrono::steady_clock::time_point end() const;

	std::chrono::steady_clock::time_point m_start, m_stop;
	bool m_running;
};

steadyTimer::steadyTimer() : m_running(false) {}
void steadyTimer::start() {
	m_running = true;
	m_start = std::chrono::steady_clock::now();
}

void steadyTimer::stop() {
	m_running = false;
	m_stop = std::chrono::steady_clock::now();
}

void steadyTimer::reset() {
	m_running = false;
	m_start = {};
	m_stop = {};
}

void steadyTimer::restart() {
	reset();
	start();
}

bool steadyTimer::running() const {return m_running;}

std::chrono::steady_clock::time_point steadyTimer::end() const {  //return stop point either the stopped time or the time rn to allow multiple checkpoints throughout a test
	return m_running ? std::chrono::steady_clock::now() : m_stop;
}

double steadyTimer::seconds() const {
	return std::chrono::duration<double>(end() - m_start).count();
}

double steadyTimer::milliseconds() const {
	return std::chrono::duration<double, std::milli>(end() - m_start).count();
}

double steadyTimer::microseconds() const {
	return std::chrono::duration<double, std::micro>(end() - m_start).count();
}

class scopeTimer : public steadyTimer {
public:
	static void selfTest();
	explicit scopeTimer();
	explicit scopeTimer(const char* p_label, std::ostream &p_ostream = std::clog);
	~scopeTimer() override;

private:
	const char* m_label;
	std::ostream& m_ostream;
};

scopeTimer::scopeTimer() : scopeTimer("Scope Timer", std::clog) {}
scopeTimer::scopeTimer(const char* p_label, std::ostream &p_ostream) : m_label(p_label), m_ostream(p_ostream) {
	start();
}

void scopeTimer::selfTest() {
	scopeTimer benchmark("Self Test Time", std::clog);
	{
		scopeTimer dummy("Dummy Timer", std::clog);
	}
}

scopeTimer::~scopeTimer() {
	stop();
	m_ostream << m_label << ": " << microseconds() << " us\n";
}