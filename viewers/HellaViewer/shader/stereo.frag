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

float checkersTextureGradBox( in vec2 p, in vec2 ddx, in vec2 ddy )
{ // credits: http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
    // filter kernel
    vec2 w = max(abs(ddx), abs(ddy)) + 0.01;  
    // analytical integral (box filter)
    vec2 i = 2.0*(abs(fract((p-0.5*w)/2.0)-0.5)-abs(fract((p+0.5*w)/2.0)-0.5))/w;
    // xor pattern
    return 0.5 - 0.5*i.x*i.y;                  
}

float checkersTexture( in vec2 p )
{ // credits: http://iquilezles.org/www/articles/checkerfiltering/checkerfiltering.htm
    vec2 q = floor(p);
    return mod( q.x+q.y, 2.0 );            // xor pattern
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
		vec2 checkersUv = (eyepos + eyedir * dist).xz;	
		if(sqrt(checkersUv.x*checkersUv.x + checkersUv.y*checkersUv.y) > 25) return vec4(.1f);
		vec2 ddxCheckersUv = dFdx( checkersUv ); 
        vec2 ddyCheckersUv = dFdy( checkersUv ); 
		return vec4(vec3(checkersTextureGradBox(checkersUv, ddxCheckersUv, ddyCheckersUv)), 1.0f);
		//return vec4(0.15f,0.15f,0.15f,1.0f);
		//return vec4(checkersTexture(checkersUv));
	}
	else // background color
	{
		return vec4(vec3(.1f), 1.0f);
	}

}

void main(void)
{
	outColor = fragmentColor();
}
