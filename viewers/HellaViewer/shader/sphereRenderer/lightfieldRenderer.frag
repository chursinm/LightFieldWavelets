#version 400 core

uniform vec3 viewspaceLightPosition;

in vec3 viewspaceVertex;
in vec2 uv;
in vec3 lightfield;

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
	outColor = vec4(lightfield, 0.5f);
}
