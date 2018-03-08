#version 400 core

uniform vec3 viewspaceLightPosition;
uniform sampler2D debugTexture;
uniform float alphaOut;
uniform float alphaMult;

in vec3 edgeDistance;
in vec3 viewspaceVertex;
in vec2 uv;

layout(location = 0) out vec4 outColor;

// constant as uniform for random performance reasons
uniform vec3 COLORS[ 12 ] = {	vec3( 1.0, 0.0, 0.0 ),
                                vec3( 0.0, 1.0, 0.0 ),
                                vec3( 0.0, 0.0, 1.0 ),
                                vec3( 1.0, 1.0, 0.0 ),
                                vec3( 0.0, 1.0, 1.0 ),
                                vec3( 1.0, 0.0, 1.0 ),

								vec3( 1.0, 0.4, 0.4 ),
                                vec3( 0.4, 1.0, 0.4 ),
                                vec3( 0.4, 0.4, 1.0 ),
                                vec3( 1.0, 1.0, 0.4 ),
                                vec3( 0.4, 1.0, 1.0 ),
                                vec3( 1.0, 0.4, 1.0 )  };


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

bool checkerboard(vec2 a, int level)
{
	return mod(a.x*level*2, 2) >= 1 != mod(a.y*level*2, 2) >= 1;
}

vec3 highlightEdge(vec3 baseColor, vec3 edgeColor)
{
	float minEdgeDistance = min(edgeDistance.x, min(edgeDistance.y, edgeDistance.z));

	// calculate a scale: smaller triangles have a bigger edge relative to their size (naive approach in screen space)
	vec3 dx_edgeDistance = dFdx(edgeDistance);
    vec3 dy_edgeDistance = dFdy(edgeDistance);
	float edgeScale = max(dot(dy_edgeDistance, dy_edgeDistance), dot(dx_edgeDistance, dx_edgeDistance));
	float edgeBound = 0.1 * pow(edgeScale, 0.1);

	// blending
	float edgeLerpFactor = smoothstep(edgeBound, 0.00, minEdgeDistance);
	return mix(baseColor, edgeColor, edgeLerpFactor);
}

vec3 highlightEdge2(vec3 baseColor, vec3 edgeColor)
{
	float minEdgeDistance = min(abs(edgeDistance.x - 0.5f), min(abs(edgeDistance.y - 0.5f), abs(edgeDistance.z - 0.5f)));
	//float minEdgeDistance = min(abs(edgeDistance.x - 0.5f), min(edgeDistance.y, edgeDistance.z));

	// calculate a scale: smaller triangles have a bigger edge relative to their size (naive approach in screen space)
	vec3 dx_edgeDistance = dFdx(edgeDistance);
    vec3 dy_edgeDistance = dFdy(edgeDistance);
	float edgeScale = max(dot(dy_edgeDistance, dy_edgeDistance), dot(dx_edgeDistance, dx_edgeDistance));
	float edgeBound = 0.01 * pow(edgeScale, 0.1);

	// blending
	float edgeLerpFactor = smoothstep(edgeBound, 0.00, minEdgeDistance);
	return mix(baseColor, edgeColor, edgeLerpFactor);
}

void main(void)
{
	int primitiveMod = gl_PrimitiveID % 12;
	vec3 color = COLORS[primitiveMod] * 0.7;

	vec3 viewDir = normalize(-viewspaceVertex);
	vec3 lightDir = normalize(viewspaceLightPosition - viewspaceVertex);
	vec3 normal = normalize(cross(dFdx(viewspaceVertex), dFdy(viewspaceVertex)));
	outColor = vec4(phong(viewDir, lightDir, normal, vec3(0), color, color, 0.5f), alphaOut);
	//outColor = texture(debugTexture, uv);
	//outColor = vec4(checkerboard(uv, 1));
	//outColor = vec4(uv, 0, 1);

	outColor.xyz = highlightEdge(outColor.xyz, vec3(0.1));
	outColor.xyz = highlightEdge2(outColor.xyz, vec3(0,0.4,0.4));
	outColor.xyz *= alphaMult;
}
