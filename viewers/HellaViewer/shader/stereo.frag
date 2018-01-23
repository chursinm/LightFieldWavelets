#version 400 compatibility
in vec4 vertex;
uniform uint eye;
uniform mat4 ivp;
uniform vec3 eyepos;


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

void main(void)
{
	vec3 planepos = vec3(0.f,-1.f,0.f);
	vec3 planedir = vec3(0.f,1.f,0.f);

	vec4 worldVertex = ivp * vertex;
	vec3 worldVertex3 = worldVertex.xyz / worldVertex.w;
	vec3 eyedir = normalize(worldVertex3 - eyepos);

	float dist = 0.f;
	bool intersectsplane = intersectRayPlane(eyepos, eyedir, planepos, planedir, dist);

	if(intersectsplane) // checkerboard
	{
		gl_FragColor = vec4(checkerboard(eyepos + eyedir * dist));
	}
	else // background color
	{
		gl_FragColor = vec4(.1f);
	}
}
