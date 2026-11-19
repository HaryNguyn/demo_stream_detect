// No browser dependency: exercise pointer coordinates and Streamlit message protocol.
const fs = require("node:fs");
const vm = require("node:vm");
const assert = require("node:assert/strict");
const posts = [], handlers = {}, windowHandlers = {}, elements = {};
const context = new Proxy({}, {get: () => () => {}});
for (const id of ["canvas", "message", "mode", "undo", "reset", "save", "clear"]) {
  elements[id] = {
    value: id === "mode" ? "rectangle" : "",
    addEventListener: (name, fn) => { handlers[name] = fn; },
    getContext: () => context,
    getBoundingClientRect: () => ({left: 10, top: 20, width: 550, height: 309.375}),
    setPointerCapture: () => {},
    hasPointerCapture: () => true,
    releasePointerCapture: () => {},
  };
}
const parent = {postMessage: value => posts.push(JSON.parse(JSON.stringify(value)))};
class Image {
  constructor() { this.complete = true; this.naturalWidth = 1920; this.naturalHeight = 1080; }
  set src(value) { this.onload(); }
}
const window = {parent, addEventListener: (name, fn) => { windowHandlers[name] = fn; }};
const html = fs.readFileSync("src/vehicle_ai/roi_component/index.html", "utf8");
vm.runInNewContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], {
  window, Image, document: {body: {scrollHeight: 700}, getElementById: id => elements[id]},
  Math, Date,
});
function render(id, points=[]) {
  windowHandlers.message({source: parent, data: {
    type: "streamlit:render", args: {context: id, points, image: "test-image"},
  }});
}
function pointer(x, y) { return {clientX: 10 + 550*x, clientY: 20 + 309.375*y, pointerId: 1}; }
function result() { return posts.filter(p => p.type === "streamlit:setComponentValue").at(-1).value; }
assert(posts.some(p => p.type === "streamlit:componentReady" && p.apiVersion === 1));
render("video1/frame1");
handlers.pointerdown(pointer(.1, .2));
handlers.pointermove(pointer(.8, .9));
handlers.pointerup(pointer(.8, .9));
elements.save.onclick();
assert.equal(result().context, "video1/frame1");
const rectangle = [[.1, .2], [.8, .2], [.8, .9], [.1, .9]];
result().points.forEach((p,i)=>p.forEach((v,j)=>assert(Math.abs(v-rectangle[i][j])<1e-9)));
render("video2/frame1", [[0,0],[1,0],[1,1]]);
elements.save.onclick();
assert.deepEqual(result().points, [[0,0],[1,0],[1,1]]);
elements.clear.onclick();
assert.deepEqual(result().points, []);
elements.mode.value = "polygon";
handlers.pointerdown(pointer(.1,.1));
handlers.pointerdown(pointer(.9,.1));
handlers.pointerdown(pointer(.5,.8));
elements.save.onclick();
assert.equal(result().points.length, 3);
assert.equal(result().context, "video2/frame1");
console.log("ROI canvas: rectangle, polygon, scaling, save/clear and video isolation passed.");
