#version 400 compatibility
in vec4 vertex;
//in vec4 worldVertex;
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

vec4 fragmentColor(vec4 clipspaceVertex)
{
	vec3 planepos = vec3(0.f,-1.f,0.f);
	vec3 planedir = vec3(0.f,1.f,0.f);

	// TODO interpolate worldVertex from vertexShader
	vec4 worldVertex = ivp * clipspaceVertex;
	vec3 worldVertex3 = worldVertex.xyz / worldVertex.w;
	vec3 eyedir = normalize(worldVertex3 - eyepos);

	float dist = 0.f;
	bool intersectsplane = intersectRayPlane(eyepos, eyedir, planepos, planedir, dist);

	
	if(intersectsplane) // checkerboard
	{
		float dotSize = 0.001f / dist;
		if(dot(clipspaceVertex.xy,clipspaceVertex.xy) < dotSize) // targetting dot
		{
			return vec4(1.f,0.f,0.f,1.f);
		}
		return vec4(checkerboard(eyepos + eyedir * dist));
	}
	else // background color
	{
		return vec4(.1f);
	}

}

void main(void)
{
	// interpolation
	// width, height
	vec4 viewportOffset = vec4(1.f/1512.f, 1.f/1680.f, 0.f, 0.f);

	vec4 accumulator = vec4(0.f);
	accumulator += fragmentColor(vertex + viewportOffset);
	accumulator += fragmentColor(vertex - viewportOffset);
	accumulator += fragmentColor(vertex + (viewportOffset * vec4(-1.f, 1.f, 0.f, 0.f)));
	accumulator += fragmentColor(vertex + (viewportOffset * vec4(1.f, -1.f, 0.f, 0.f)));
	accumulator /= 4.f;


	gl_FragColor = fragmentColor(vertex);
}
