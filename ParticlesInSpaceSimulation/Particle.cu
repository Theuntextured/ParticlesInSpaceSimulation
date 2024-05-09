#include "Particle.cuh"

Particle::Particle(float xMax, float yMax) {
	location = sf::Vector2f(rnd(0, xMax), rnd(0, yMax));
	//velocity = sf::Vector2f(rnd(-10, 10), rnd(-10, 10));
	velocity = sf::Vector2f(0, 0);
	color = sf::Color(rnd(0, 255), rnd(0, 255), rnd(0, 255), 255);
	mass = 1;
	protons = 1;
	neutrons = 0;
	radius = 1;
	exists = true;
}
Particle::Particle() {
	exists = false;
	mass = 0;
	neutrons = 0;
	protons = 0;
	radius = 0;
}

float rnd(float LO, float HI) {
	return LO + static_cast <float> (rand()) / (static_cast <float> (RAND_MAX / (HI - LO)));
}

__device__ void handleVelocity(float dt, Particle* particle) {
	particle->location.x += particle->velocity.x * dt;
	particle->location.y += particle->velocity.y * dt;
}

__device__ void drawParticle(Particle* p, sf::Uint8* pixelBuffer, sf::VideoMode vm) {
	int pos;
	for (int x = p->location.x - p->radius; x < p->location.x + p->radius; x++) {
		for (int y = p->location.y - p->radius; y < p->location.y + p->radius; y++) {
			if (x >= 0 && y >= 0 && x < vm.width && y < vm.height && (p->radius * p->radius) >= (y - p->location.y) * (y - p->location.y) + (x - p->location.x) * (x - p->location.x)) {
				pos = (vm.width * y + x) * 4;
				pixelBuffer[pos + 0] = p->color.r;
				pixelBuffer[pos + 1] = p->color.g;
				pixelBuffer[pos + 2] = p->color.b;
				pixelBuffer[pos + 3] = p->color.a;
			}
		}
	}
}

__device__ void updateParticleProperties(Particle* p) {
	p->mass = p->protons + p->neutrons;
	p->radius = sqrt(float(p->mass));
}

__device__ void handleForces(Particle* particles, float dt, int p, int particleCount) {
	float dist_squared;
	float dist;
	float d6;
	float ax;
	float ay;
	float sigma;
	float LJPMagnitude;

	for (int i = 0; i < particleCount; i++) {
		if (i != p && particles[i].exists) {
			dist_squared = (particles[i].location.x - particles[p].location.x) * (particles[i].location.x - particles[p].location.x) + (particles[i].location.y - particles[p].location.y) * (particles[i].location.y - particles[p].location.y);
			dist = sqrt(dist_squared);
			d6 = dist_squared * dist_squared * dist_squared;
			sigma = particles[i].radius + particles[p].radius;
			sigma = sigma * sigma * sigma * sigma * sigma * sigma;
			LJPMagnitude = LJPScale * ((sigma * sigma) / (d6 * d6 * dist) - 0.5 * (sigma / (d6 * dist)));

			ax = (particles[i].location.x - particles[p].location.x) * ((gravitationalConstant * particles[p].mass) / dist_squared / dist - LJPMagnitude);
			ay = (particles[i].location.y - particles[p].location.y) * ((gravitationalConstant * particles[p].mass) / dist_squared / dist - LJPMagnitude);
			particles[p].velocity.x += ax * dt;
			particles[p].velocity.y += ay * dt;
		}
	}
}

__global__ void ParticleTick(float dt, Particle* particles, sf::Uint8* pixelBuffer, int particleCount, sf::VideoMode vm, bool draw)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > particleCount) return;
	if (!particles[i].exists) return;

	handleForces(particles, dt, i, particleCount);
	handleVelocity(dt, &particles[i]);
	if (!draw) return;
	drawParticle(&particles[i], pixelBuffer, vm);
}
