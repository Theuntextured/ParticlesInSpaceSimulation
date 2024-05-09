#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <SFML/Graphics.hpp>
#include <cstdlib>
#include "Settings.cuh"

class Particle {
public:
	Particle(float xMax, float yMax);
	Particle();

	sf::Vector2f location;
	sf::Vector2f velocity;
	int mass;
	int protons;
	int neutrons;
	sf::Color color;
	float radius;
	bool exists;
};

float rnd(float LO, float HI);

__global__ void ParticleTick(float dt, Particle* particles, sf::Uint8* pixelBuffer, int particleCount, sf::VideoMode vm, bool draw);