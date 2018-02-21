#include "stdafx.h"
#include "Blit.h"


void Blit::render()
{
	glBindVertexArray(m_vao);
	glDrawArrays(GL_TRIANGLES, 0, 3);
}

void Blit::generateTriangle()
{
	// Create blitting VAO 
	const GLfloat fullscreenTriangle[] = {
		-1.0f, 1.0f,  0.0f, 1.0f,
		-1.0f, -3.0f, 0.0f, -1.0f,
		3.0f,  1.0f,  2.0f, 1.0f
	};

	glGenBuffers(1, &m_vb);
	glBindBuffer(GL_ARRAY_BUFFER, m_vb);
	{
		glBufferData(GL_ARRAY_BUFFER, sizeof(fullscreenTriangle), &fullscreenTriangle[0], GL_STATIC_DRAW);
	}
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glGenVertexArrays(1, &m_vao);
	glBindVertexArray(m_vao);
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vb);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, 0);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(GLfloat) * 4, (void*)(sizeof(GLfloat)*2));
		glEnableVertexAttribArray(1);
	}
	glBindVertexArray(0);
}

Blit::Blit()
{
	generateTriangle();
}


Blit::~Blit()
{
	glDeleteBuffers(1, &m_vb);
	glDeleteVertexArrays(1, &m_vb);
}
