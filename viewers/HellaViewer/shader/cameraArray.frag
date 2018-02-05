#version 460 compatibility

uniform sampler2D cameraImage;

uniform mat4 mvp;
uniform uvec2 cameraGridDimension;
uniform uint quadID;
uniform mat4 ivp;
uniform vec3 worldspaceFocalPlanePosition;
uniform vec3 worldspaceFocalPlaneDirection;
uniform vec3 worldspaceEyePosition;
uniform float focalPlaneDistance;

in vec2 uv;
sample in vec4 clipspaceCameraplaneVertex;
flat in vec4 clipspaceCameraplaneCamera;


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
	float y = worldIntersection.y;

	return mod(x, 10) >= 5 != mod(y, 10) >= 5;
}

vec2 cameraUV(vec3 focalPlaneIntersection, vec4 clipspaceCameraplaneCamera)
{
	vec4 worldSpaceCameraplaneCamera = (ivp  * clipspaceCameraplaneCamera);
	vec3 worldSpaceCameraplaneCamera3 = worldSpaceCameraplaneCamera.xyz / worldSpaceCameraplaneCamera.w;
	focalPlaneIntersection = focalPlaneIntersection - worldSpaceCameraplaneCamera3;
	return vec2(focalPlaneDistance * focalPlaneIntersection.x / focalPlaneIntersection.z, focalPlaneDistance * focalPlaneIntersection.y / focalPlaneIntersection.z);
}

void main(void)
{
	vec4 worldVertex = ivp * clipspaceCameraplaneVertex;
	vec3 worldVertex3 = worldVertex.xyz;
	if(worldVertex.w != 0.f) worldVertex3 /= worldVertex.w;
	vec3 worldSpaceEyeDirection = normalize(worldVertex3 - worldspaceEyePosition);

	float dist = 0.f;
	bool intersectsplane = intersectRayPlane(worldspaceEyePosition, worldSpaceEyeDirection, worldspaceFocalPlanePosition, worldspaceFocalPlaneDirection, dist);

	
	if(intersectsplane) 
	{
		vec3 focalPlaneIntersection = worldspaceEyePosition + worldSpaceEyeDirection * dist;
		//gl_FragColor = vec4(checkerboard(focalPlaneIntersection));
		//gl_FragColor = vec4(uv,0,0);
		gl_FragColor = texture(cameraImage, cameraUV(focalPlaneIntersection, clipspaceCameraplaneCamera));
	}
	else // background color
	{
		gl_FragColor = vec4(1,0,0,0); // error
	}

    //gl_FragColor = vec4(uv,0.f,1.f);
	//gl_FragColor = texture(cameraImage, uv);
	//gl_FragColor = vec4(1,0,0,0);
}
