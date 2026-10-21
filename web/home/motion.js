// Entrance motion: masked headline on load, staggered reveals as sections enter. Content stays visible without script.
export function startMotion(){
  const root = document.documentElement;
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
  root.classList.add("motion");
  requestAnimationFrame(() => root.classList.add("loaded"));

  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (!entry.isIntersecting) return;
      entry.target.classList.add("in");
      observer.unobserve(entry.target);
    });
  }, { rootMargin: "0px 0px -10% 0px" });

  const watch = () => document.querySelectorAll(".reveal:not(.in)").forEach((node, index) => {
    node.style.setProperty("--delay", `${(index % 4) * 80}ms`);
    observer.observe(node);
  });
  watch();
  return watch;
}
