// Hero reel: fills the case frames and advances one frame per stretch of scroll while the stage is pinned.
import { sequenceFrame, scoreFrame } from "./case-view.js";

export function fillReel({ cases, content }){
  const record = cases.find(item => item.firm === content.hero_case) || cases[0];
  document.getElementById("frameSequence").innerHTML = sequenceFrame(record);
  document.getElementById("frameScore").innerHTML = scoreFrame(record, content.outcome_labels);
}

export function startReel(){
  const hero = document.querySelector(".hero");
  const track = document.getElementById("reelTrack");
  const frames = [...track.children];
  let current = -1;

  function update(){
    const travel = hero.offsetHeight - window.innerHeight;
    const progress = travel > 0 ? Math.min(1, Math.max(0, -hero.getBoundingClientRect().top / travel)) : 0;
    const step = Math.min(frames.length - 1, Math.floor(progress * frames.length));
    if (step === current) return;
    current = step;
    track.style.setProperty("--step", step);
    frames.forEach((frame, index) => frame.classList.toggle("active", index === step));
  }

  window.addEventListener("scroll", () => requestAnimationFrame(update), { passive: true });
  window.addEventListener("resize", update);
  update();
}
