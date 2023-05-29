#version 330 core
layout(location = 0) in vec3 vertexPositionMS;
layout(location = 1) in vec3 velocity;

uniform mat4 MVP;
out float color;



void main() {
    //magnitude of the velocity
    color = sqrt(velocity.x * velocity.x + velocity.y * velocity.y + velocity.z * velocity.z);
    color = 1 / (0.01 * color);
    gl_PointSize = 3.0;
    gl_Position = MVP * vec4(vertexPositionMS, 1);
}