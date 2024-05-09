#pragma once
#include "ViewportManager.cuh"
#include <SFML/System.hpp>
#include <iostream>
#include <ctime>
#include "Particle.cuh"
#include "Settings.cuh"

class Engine {
public:
	Engine();
	~Engine();
	void Tick();
private:
	sf::Clock dtClock;
	float dt;
	Particle* particles;
	Particle* d_particles;
	ViewportManager* vm;
};