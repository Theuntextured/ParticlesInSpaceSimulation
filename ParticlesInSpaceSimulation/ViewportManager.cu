#include "ViewportManager.cuh"

ViewportManager::ViewportManager(int sizeX, int sizeY) {
	screenSize = sf::VideoMode(sizeX, sizeY);
	window.create(screenSize, "Sim");

	pixelBuffer = new sf::Uint8[sizeX * sizeY * 4];
	for (int i = 0; i < sizeX * sizeY * 4; i++) {
		pixelBuffer[i] = 255;
	}
	cudaMalloc(&d_pixelBuffer, sizeof(sf::Uint8) * sizeX * sizeY * 4);
	cudaMemcpy(d_pixelBuffer, pixelBuffer, sizeof(sf::Uint8) * sizeX * sizeY * 4, cudaMemcpyHostToDevice);
	screenTexture.create(sizeX, sizeY);
	screenTexture.setSmooth(true);

	font.loadFromFile("arial.ttf");
	fpsText.setFont(font);
	fpsText.setCharacterSize(11);
	fpsText.setFillColor(sf::Color::Black);
	fpsText.setOutlineColor(sf::Color::White);
	fpsText.setOutlineThickness(1.0);
}

ViewportManager::~ViewportManager()
{
	cudaFree(d_pixelBuffer);
}

__device__ int biToSingleDimentionIndex(int x, int y, int width) {
	return width * y + x;
}

__global__ void clearBuffer(sf::Uint8* buffer, int pixelCount, float dt, sf::VideoMode vm)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i > pixelCount) return;
	float sum;
	int items;
	int ox = i % vm.width;
	int oy = i / vm.width;

	for (int n = 0; n < 4; n++) {
		sum = 0;
		items = 0;
		for (int x = ox-1; x < ox + 2; x++) {
			for (int y = oy - 1; y < oy + 2; y++) {
				if (x >= 0 && y >= 0 && x < vm.width && y < vm.height && !(y == 0 && x == 0)) {
					sum += buffer[biToSingleDimentionIndex(x, y, vm.width) * 4 + n];
					items++;
				}
			}
		}
		buffer[i * 4 + n] = sum / items;
	}
	return;
	i *= 4;
	buffer[i + 0] *= pow(0.5, dt);
	buffer[i + 1] *= pow(0.5, dt);
	buffer[i + 2] *= pow(0.5, dt);
	buffer[i + 3] *= pow(0.5, dt);
	
}

bool ViewportManager::tick(float dt)
{

	sf::Event event;
	while (window.pollEvent(event))
	{
		// "close requested" event: we close the window
		if (event.type == sf::Event::Closed)
			window.close();
	}

	cudaMemcpy(pixelBuffer, d_pixelBuffer, sizeof(sf::Uint8) * screenSize.width * screenSize.height * 4, cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	screenTexture.update(pixelBuffer);
	screenSprite.setTexture(screenTexture);

	window.clear();
	screenSprite.setColor(sf::Color::White);
	window.draw(screenSprite);
	fpsText.setString(std::to_string(int(1 / dt)));
	window.draw(fpsText);
	window.display();
	clearBuffer << < (screenSize.width * screenSize.height + 1023) / 1024, 1024 >> > (d_pixelBuffer, screenSize.width * screenSize.height, dt, screenSize);
	cudaDeviceSynchronize();

	return window.isOpen();
}