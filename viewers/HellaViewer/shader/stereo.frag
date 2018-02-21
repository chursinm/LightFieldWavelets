#version 400 core

uniform mat4 ivp;
uniform vec3 eyepos;

noperspective in vec3 worldVertex3;

layout(location = 0) out vec4 outColor;

bool intersectRayPlane(
    in vec3 ray_origin,
    in vec3 ray_direction,
    in vec3 center,
    in vec3 normal,
    out float t)
{
    float denom = dot(ray_direction, normal);
    if (denom == 0.f) {
        return false;
    }
    t = dot(center - ray_origin, normal) / denom;
    if (t > 1e-4f) {
        return true;
    }
    return false;
}

bool checkerboard(vec3 worldIntersection)
{
	//2D
	float x = worldIntersection.x;
	float y = worldIntersection.z;

	return mod(x, 10) >= 5 != mod(y, 10) >= 5;
}

vec4 fragmentColor()
{
	vec3 planepos = vec3(0.f,-1.f,0.f);
	vec3 planedir = vec3(0.f,1.f,0.f);

	vec3 eyedir = normalize(worldVertex3 - eyepos);

	float dist = 0.f;
	bool intersectsplane = intersectRayPlane(eyepos, eyedir, planepos, planedir, dist);

	
	if(intersectsplane) // checkerboard
	{
		return vec4(checkerboard(eyepos + eyedir * dist));
	}
	else // background color
	{
		return vec4(.1f);
	}

}

void main(void)
{
	outColor = fragmentColor();
}
