#pragma once

#include <iostream>

#include "../utils/matrix.hh"

class Layer {
protected:
	std::string name;

public:
	virtual ~Layer() = 0;

	virtual Matrix& forward(Matrix& A) = 0;
	virtual Matrix& backprop(Matrix& dZ, float learning_rate) = 0;

	std::string getName() { return this->name; };

};

inline Layer::~Layer() {}
