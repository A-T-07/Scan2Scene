// Scan2Scene viewer v2 -- presentation mode by default (light background, RGB points, no trajectory).
// Loads the existing, unchanged export ../final_phone_with_camera_trajectory.glb (point cloud + camera lines).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const GLB = "../final_phone_with_camera_trajectory.glb";
const BG = { light: 0xeef1f4, dark: 0x0b0e12 };
const $ = (id) => document.getElementById(id);
const status = (t) => ($("status").textContent = t);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setSize(innerWidth, innerHeight);
renderer.outputColorSpace = THREE.LinearSRGBColorSpace;   // COLOR_0 = original image RGB; show unchanged
document.body.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(BG.light);
const camera = new THREE.PerspectiveCamera(60, innerWidth / innerHeight, 0.01, 200);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;

// round point sprite (soft-edged disc) -> circular points instead of squares
function discTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), grd = g.createRadialGradient(32, 32, 0, 32, 32, 32);
  grd.addColorStop(0, "rgba(255,255,255,1)"); grd.addColorStop(0.7, "rgba(255,255,255,1)"); grd.addColorStop(1, "rgba(255,255,255,0)");
  g.fillStyle = grd; g.fillRect(0, 0, 64, 64);
  const t = new THREE.CanvasTexture(c); t.colorSpace = THREE.NoColorSpace; return t;
}
const pointMat = new THREE.PointsMaterial({ size: 0.013, sizeAttenuation: true, vertexColors: true,
  map: discTexture(), alphaTest: 0.5, transparent: false });

const groups = { path: null, frustums: null, markers: null };
let box = null, floorY = 0, ceilY = 3, points = null;

// split the single LINES primitive of the export into path / frustums / markers by vertex colour
function splitLines(obj) {
  const pos = obj.geometry.attributes.position, col = obj.geometry.attributes.color;
  const buckets = { path: [[], []], frustums: [[], []], markers: [[], []] };
  const c = new THREE.Color();
  for (let i = 0; i < pos.count; i += 2) {
    const r = col.getX(i), gr = col.getY(i), b = col.getZ(i);
    const key = (r > 0.9 && b > 0.9) ? "path" : (r > 0.9 && gr > 0.8 && b < 0.1) ? "frustums" : "markers";
    for (const k of [i, i + 1]) {
      buckets[key][0].push(pos.getX(k), pos.getY(k), pos.getZ(k));
      buckets[key][1].push(col.getX(k), col.getY(k), col.getZ(k));
    }
  }
  for (const [key, [p, cl]] of Object.entries(buckets)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.Float32BufferAttribute(p, 3));
    g.setAttribute("color", new THREE.Float32BufferAttribute(cl, 3));
    const ls = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ vertexColors: true }));
    ls.matrixAutoUpdate = false; ls.matrix.copy(obj.matrixWorld); ls.visible = false;
    scene.add(ls); groups[key] = ls;
  }
  obj.visible = false;
}

function setView(kind) {
  if (!box) return;
  const c = box.getCenter(new THREE.Vector3()), s = box.getSize(new THREE.Vector3());
  const r = Math.max(s.x, s.z);
  if (kind === "top") {
    controls.target.set(c.x, floorY, c.z);
    camera.position.set(c.x, floorY + r * 1.35, c.z + 0.001);
  } else if (kind === "overview") {   // elevated oblique view into the room (ceiling band clipped for display)
    controls.target.set(c.x, floorY + 0.25 * (ceilY - floorY), c.z);
    camera.position.set(c.x + r * 0.28, ceilY + r * 0.38, c.z + r * 0.42);
  } else {   // inside: stand at one end of the room's long axis, above head height, look down along it
    const alongX = s.x >= s.z;
    const d = alongX ? new THREE.Vector3(1, 0, 0) : new THREE.Vector3(0, 0, 1);
    const half = (alongX ? s.x : s.z) * 0.42;
    camera.position.set(c.x - d.x * half, floorY + 0.85 * (ceilY - floorY), c.z - d.z * half);
    controls.target.set(c.x + d.x * half * 0.4, floorY + 0.3 * (ceilY - floorY), c.z + d.z * half * 0.4);
  }
  camera.near = r / 2000; camera.far = r * 30; camera.updateProjectionMatrix(); controls.update();
}

function analysis(on) {
  $("analysis").classList.toggle("show", on);
  $("mode").classList.toggle("on", on);
  $("mode").textContent = on ? "Presentation mode" : "Analysis mode";
  if (points) status(`${points.geometry.attributes.position.count.toLocaleString()} points · ${on ? "analysis" : "presentation"} mode`);
  if (groups.path) {
    groups.path.visible = on && $("showPath").checked;
    groups.markers.visible = on && $("showMarkers").checked;
    groups.frustums.visible = on && $("showFrustums").checked;
  }
}

new GLTFLoader().load(GLB, (gltf) => {
  scene.add(gltf.scene);
  gltf.scene.updateMatrixWorld(true);
  let n = 0;
  gltf.scene.traverse((o) => {
    if (o.isPoints) { o.material = pointMat; points = o; n += o.geometry.attributes.position.count; }
  });
  const lines = []; gltf.scene.traverse((o) => { if (o.isLine || o.isLineSegments) lines.push(o); });
  lines.forEach(splitLines);
  box = new THREE.Box3().setFromObject(points);
  // floor / ceiling from robust height percentiles of the (display-transformed) cloud
  const p = points.geometry.attributes.position, m = points.matrixWorld, v = new THREE.Vector3(), ys = [];
  for (let i = 0; i < p.count; i += 97) { v.fromBufferAttribute(p, i).applyMatrix4(m); ys.push(v.y); }
  ys.sort((a, b) => a - b);
  floorY = ys[Math.floor(ys.length * 0.02)]; ceilY = ys[Math.floor(ys.length * 0.98)];
  // display-only clipping of the ceiling band so the room interior is visible from above (data unchanged)
  pointMat.clippingPlanes = [new THREE.Plane(new THREE.Vector3(0, -1, 0), ceilY - 0.12 * (ceilY - floorY))];
  renderer.localClippingEnabled = true;
  setView("overview");
  analysis(false);
  status(`${n.toLocaleString()} points · presentation mode`);
}, (e) => e.total && status(`loading… ${Math.round(100 * e.loaded / e.total)} %`),
   () => status("Could not load the GLB. Start the viewer with open_viewer.bat (it serves the files locally)."));

$("size").oninput = (e) => { pointMat.size = parseFloat(e.target.value); };
$("bgLight").onclick = () => { scene.background.set(BG.light); $("bgLight").classList.add("on"); $("bgDark").classList.remove("on"); };
$("bgDark").onclick = () => { scene.background.set(BG.dark); $("bgDark").classList.add("on"); $("bgLight").classList.remove("on"); };
$("viewInside").onclick = () => setView("inside");
$("viewOverview").onclick = () => setView("overview");
$("viewTop").onclick = () => setView("top");
$("mode").onclick = () => analysis(!$("analysis").classList.contains("show"));
["showPath", "showMarkers", "showFrustums"].forEach((id) => ($(id).onchange = () => analysis(true)));
$("toggle").onclick = () => { $("panel").classList.toggle("collapsed"); $("toggle").textContent = $("panel").classList.contains("collapsed") ? "+" : "–"; };
addEventListener("resize", () => { camera.aspect = innerWidth / innerHeight; camera.updateProjectionMatrix(); renderer.setSize(innerWidth, innerHeight); });
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });
