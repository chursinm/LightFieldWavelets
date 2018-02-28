#version 400 core

uniform vec3 viewspaceLightPosition;

in vec3 viewspaceVertex;

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
	int primitiveMod = gl_PrimitiveID % 3;
	float r = float(primitiveMod == 0);
	float g = float(primitiveMod == 1);
	float b = float(primitiveMod == 2);
	vec3 color = vec3(r,g,b);

	vec3 viewDir = normalize(-viewspaceVertex);
	vec3 lightDir = normalize(viewspaceLightPosition - viewspaceVertex);
	vec3 normal = normalize(cross(dFdx(viewspaceVertex), dFdy(viewspaceVertex)));
	outColor = vec4(phong(viewDir, lightDir, normal, vec3(0), color, color, 0.5f), 1.0f);
	//outColor = vec4(r,g,b,1);
}
