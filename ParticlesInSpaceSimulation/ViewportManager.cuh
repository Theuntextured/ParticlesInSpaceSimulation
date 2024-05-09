#pragma once
#include <SFML/Window.hpp>
#include <SFML/Graphics.hpp>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <string>

class ViewportManager {
public:
	ViewportManager(int sizeX, int sizeY);
	~ViewportManager();
	bool tick(float dt);

	sf::RenderWindow window;
	sf::Uint8* d_pixelBuffer;

private:
	sf::Uint8* pixelBuffer;
	sf::VideoMode screenSize;
	sf::Texture screenTexture;
	sf::Sprite screenSprite;
	sf::Text fpsText;
	sf::Font font;
};


__global__ void clearBuffer(sf::Uint8* buffer, int pixelCount, float dt);