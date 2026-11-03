// Gemeinsame Alterung aller Objekte, nach dem Vorbild des Wano-Musterhauses: verhaltene Farben, Regenspuren an senkrechten
// Flächen, Staub auf waagerechten, dunkle Unterseiten. Alles in Weltkoordinaten, damit gleiche Modelle verschieden altern.
const grime = { value: null };
const done = new WeakSet();

export function setGrimeTexture(tex) {
  grime.value = tex;
}

export function weather(mat, { desat = 0.3, dust = 1, rough = 0 } = {}) {
  if (!mat || done.has(mat) || !mat.isMeshStandardMaterial || mat.transparent) return mat;
  done.add(mat);
  if (rough) mat.roughness = Math.max(mat.roughness, rough);
  const prev = mat.onBeforeCompile, prevKey = mat.customProgramCacheKey?.() ?? '';
  mat.onBeforeCompile = (sh, r) => {
    prev?.call(mat, sh, r);
    sh.uniforms.uGrime = grime;
    sh.uniforms.uAge = { value: [desat, dust] };
    sh.vertexShader = sh.vertexShader
      .replace('#include <common>', '#include <common>\nvarying vec3 vWxPos;\nvarying vec3 vWxNor;')
      .replace('#include <project_vertex>', `#include <project_vertex>
        {
          vec4 wxp = vec4(transformed, 1.0);
          vec3 wxn = objectNormal;
          #ifdef USE_INSTANCING
            wxp = instanceMatrix * wxp;
            wxn = mat3(instanceMatrix) * wxn;
          #endif
          vWxPos = (modelMatrix * wxp).xyz;
          vWxNor = mat3(modelMatrix) * wxn;
        }`);
    sh.fragmentShader = sh.fragmentShader
      .replace('#include <common>', '#include <common>\nuniform sampler2D uGrime;\nuniform vec2 uAge;\nvarying vec3 vWxPos;\nvarying vec3 vWxNor;')
      .replace('#include <color_fragment>', `#include <color_fragment>
        {
          vec3 wn = normalize(vWxNor);
          float along = abs(wn.x) > abs(wn.z) ? vWxPos.z : vWxPos.x;
          float streak = texture2D(uGrime, vec2(along * 0.37, vWxPos.y * 0.04)).r;
          float blot = texture2D(uGrime, (vWxPos.xz + vWxPos.y) * 0.13).g;
          float up = smoothstep(0.55, 0.95, wn.y);
          float grey = dot(diffuseColor.rgb, vec3(0.3, 0.59, 0.11));
          diffuseColor.rgb = mix(diffuseColor.rgb, vec3(grey), uAge.x);
          diffuseColor.rgb *= mix(1.0, 0.68 + 0.5 * streak, 1.0 - up);
          diffuseColor.rgb *= 0.82 + 0.36 * blot;
          diffuseColor.rgb *= 1.0 + up * uAge.y * smoothstep(0.3, 0.8, blot) * 0.35;
          diffuseColor.rgb *= mix(0.62, 1.0, smoothstep(-0.7, 0.1, wn.y));
        }`);
  };
  mat.customProgramCacheKey = () => `${prevKey}|weather`;
  mat.needsUpdate = true;
  return mat;
}
