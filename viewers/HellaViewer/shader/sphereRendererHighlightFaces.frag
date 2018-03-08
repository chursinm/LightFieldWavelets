#version 400 core

uniform vec3 viewspaceLightPosition;
uniform sampler2D debugTexture;

in vec3 edgeDistance;
in vec3 viewspaceVertex;
in vec2 uv;

layout(location = 0) out vec4 outColor;

vec3 phong(vec3 viewDir, vec3 lightDir, vec3 normal, vec3 ambient, vec3 diffuse, vec3 specular, float shininess)
{
    float lambertian = max(dot(lightDir,normal), 0.0f);
    float specularExponent = 0.0f;

    if(lambertian > 0.0) {
        vec3 halfDir = normalize(lightDir + viewDir);
        float specAngle = max(dot(halfDir, normal), 0.0f);
        specularExponent = pow(specAngle, shininess);
    }

    return vec3(ambient + diffuse * lambertian + specular * specularExponent);
}


void main(void)
{
	vec3 color = edgeDistance;

	vec3 viewDir = normalize(-viewspaceVertex);
	vec3 lightDir = normalize(viewspaceLightPosition - viewspaceVertex);
	vec3 normal = normalize(cross(dFdx(viewspaceVertex), dFdy(viewspaceVertex)));
	outColor = vec4(phong(viewDir, lightDir, normal, vec3(0), color, color, 0.5f), 0.4f);
	
	// -------------------------- render edges --------------------------------
	const vec3 edgeColor = vec3(1,1,0);
	float minEdgeDistance = min(edgeDistance.x, min(edgeDistance.y, edgeDistance.z));

	// calculate a scale: smaller triangles have a bigger edge relative to their size (naive approach in screen space)
	vec3 dx_edgeDistance = dFdx(edgeDistance);
    vec3 dy_edgeDistance = dFdy(edgeDistance);
	float edgeScale = max(dot(dy_edgeDistance, dy_edgeDistance), dot(dx_edgeDistance, dx_edgeDistance));
	float edgeBound = 0.1 * pow(edgeScale, 0.1);

	// blending
	float edgeLerpFactor = smoothstep(edgeBound, 0.00, minEdgeDistance);
	outColor.xyz = mix(outColor.xyz, edgeColor, edgeLerpFactor);
}
