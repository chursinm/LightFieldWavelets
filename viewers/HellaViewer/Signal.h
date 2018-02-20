#pragma once
#include <vector>
#include <functional>
/*
A simple Signal/Slot implementation. Lacks removal off Slots, among other things.
*/
template <typename... Parameters>
class Signal
{
public:
	/*
	Register a new callback
	*/
	void operator() (std::function<void(Parameters...)> callback)
	{
		m_slots.push_back(callback);
	}

	/*
	Call all registered callbacks
	*/
	void operator() (Parameters... params)
	{
		for(auto& c : m_slots) c(params...);
	}

private:
	std::vector<std::function<void(Parameters...)>> m_slots;
};

