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
		mSlots.push_back(callback);
	}

	/*
	Call all registered callbacks
	*/
	void operator() (Parameters... params)
	{
		for(auto& c : mSlots) c(params...);
	}

private:
	std::vector<std::function<void(Parameters...)>> mSlots;
};

