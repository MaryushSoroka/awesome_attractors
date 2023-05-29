#version 450 core

layout (location = 0) out vec4 fragColor;

uniform sampler2D starImage;
in float color;

void main() {
  //Don't ask me why I have to divide by 3 here, it has something
  //to do with the texture image I use and I don't know what it is

  fragColor = texture(starImage, gl_PointCoord / 3.0) * vec4(color, 0.2, 1 / color, 1);
  // fragColor = texture(starImage, gl_PointCoord / 3.0) * vec4(0.2, 0.2, 0.2, 1);
}