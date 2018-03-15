#pragma once
class TrackballCamera
{
public:
	TrackballCamera(unsigned int viewportWidth, unsigned int viewportHeight, double fovy = 106.);
	~TrackballCamera();
	void pan(float dx, float dy);

	void move(float dx, float dz);

	void zoom(float dx);
	void rotate(const glm::uvec2 & currentCursorPosition, const glm::uvec2 & lastCursorPosition);
	glm::mat4x4 viewMatrix() const;
	glm::mat4x4 projectionMatrix() const;
	float fovy() const;
	void reset();
	glm::vec3 getPosition() const;

private:
	glm::vec3 calculateTrackballPosition(const glm::uvec2 & point) const;

	glm::quat m_Rotation;
	glm::vec3 m_Position;
	glm::uvec2 m_ViewportSize;
	double m_fovy;
	// Some standard unit vectors
	const glm::vec3 c_Up = glm::vec3(0.f,1.f,0.f),
		c_Side = glm::vec3(1.f,0.f,0.f),
		c_Forward = glm::vec3(0.f,0.f,1.f);
};

