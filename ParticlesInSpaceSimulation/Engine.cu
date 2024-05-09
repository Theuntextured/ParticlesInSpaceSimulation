#include "Engine.cuh"


Engine::Engine() {
	vm = new ViewportManager(windowWidth, windowHeight);
	srand(static_cast <unsigned> (time(0)));
	particles = new Particle[maxParticles];
	for (int i = 0; i < maxParticles * 0.75; i++) {
		particles[i] = Particle(windowWidth, windowHeight);
	}

	cudaMalloc(&d_particles, maxParticles * sizeof(Particle));
	cudaDeviceSynchronize();
	cudaMemcpy(d_particles, particles, maxParticles * sizeof(Particle), cudaMemcpyHostToDevice);
	cudaDeviceSynchronize();
	
	delete[] particles;
	
	do {
		Tick();
	} while (vm->tick(dt));
}

Engine::~Engine()
{

}

void Engine::Tick() {
	dt = dtClock.restart().asSeconds();
	for (int i = 0; i < substepCount; i++) {
		ParticleTick << <maxParticles + 1023, 1024 >> > (dt / substepCount / timeDilation, d_particles, vm->d_pixelBuffer, maxParticles, sf::VideoMode(windowWidth, windowHeight), i == substepCount - 1);
	}
}