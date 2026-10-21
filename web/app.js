// Front-page entry: load data, render sections, start motion.
import { loadAll } from "./home/data.js";
import { bindings, applyBindings, renderTimeline } from "./home/render.js";
import { mountExplorer } from "./home/explorer.js";
import { mountStages } from "./home/stages.js";
import { mountCases } from "./home/cases.js";
import { fillReel, startReel } from "./home/reel.js";
import { startMotion } from "./home/motion.js";

const rewatch = startMotion();
startReel();

loadAll()
  .then(data => {
    const values = bindings(data);
    applyBindings(values);
    fillReel(data);
    mountStages(data.content, values);
    mountExplorer(data);
    mountCases(data);
    renderTimeline(data);
    if (rewatch) rewatch();
  })
  .catch(error => {
    document.getElementById("readout").textContent = `결과 파일을 불러오지 못했습니다: ${error.message}`;
  });
