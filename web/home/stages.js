// Method slider: one range input drives the clause on the left and the fragment on the right.
import { escapeHtml } from "./format.js";

function taxonomyFragment(content){
  return `<span class="ui-head">공시 분류 체계<span>6개 유형</span></span>
    <ul class="tax">${content.taxonomy.map(item => `
      <li><span class="chip ${item.code}">${item.code}</span><span>${escapeHtml(item.name)}<small>${escapeHtml(item.examples)}</small></span></li>`).join("")}</ul>`;
}

function featureFragment(content){
  return `<span class="ui-head">기업-연도마다 만드는 순서 변수<span>기준일까지</span></span>
    <ul class="feat">${content.features.map(text => `<li>${escapeHtml(text)}</li>`).join("")}</ul>`;
}

function splitFragment(values){
  return `<span class="ui-head">시간 분할 검증<span>1년 안 부실</span></span>
    <div class="split">
      <div class="bars"><span class="train" style="flex:7">2015–2021 학습</span><span class="test" style="flex:3">2022– 검증</span></div>
      <dl>
        <dt>학습 기업-연도</dt><dd>${values.n_train_365}</dd>
        <dt>검증 기업-연도</dt><dd>${values.n_test_365}</dd>
        <dt>검증 기간 부실</dt><dd>${values.pos_365}곳 (${values.base_rate_365})</dd>
      </dl>
    </div>`;
}

function marketFragment(values){
  return `<span class="ui-head">주가 변수<span>${values.market_features}개</span></span>
    <figure><img src="./figures/paper4_market_coverage_en.webp" alt="연도별 가격 자료 연결률 그림." width="1600" height="538" loading="lazy" />
    <figcaption>기준일 전 가격 기록이 있는 기업-연도 ${values.price_rate}. 과거 수익률, 거래량, 변동성, 낙폭, 시장 대비 수익률을 썼습니다.</figcaption></figure>`;
}

const NOTES = [
  values => `${values.event_firms}개사가 2015–2024년에 낸 주요사항보고서 ${values.event_rows}건을 보고서 이름으로 여섯 유형에 나눴습니다.`,
  () => "기업-연도마다 기준일까지 나온 공시만 써서 건수, 간격, 반복을 변수로 만들었습니다. 재무제표도 기준일까지 나온 것만 썼습니다.",
  values => `${values.split}했습니다. 1년 안 부실은 드문 사건이라 ROC AUC와 함께 상위 10% 리프트를 주된 지표로 봤습니다.`,
  values => `주가가 이미 이 정보를 담고 있는지 보려고 시장 변수 ${values.market_features}개를 더했습니다. 공시 순서 신호는 주가를 넣어도 남았습니다.`
];

export function mountStages(content, values){
  const range = document.getElementById("stageRange");
  const clause = document.getElementById("stage-clause");
  const note = document.getElementById("stageNote");
  const fragment = document.getElementById("stageFragment");
  const names = document.getElementById("stageNames");
  const builders = [() => taxonomyFragment(content), () => featureFragment(content), () => splitFragment(values), () => marketFragment(values)];
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  names.innerHTML = content.stages.map((stage, index) => `<li><button type="button" data-index="${index}">${escapeHtml(stage.name)}</button></li>`).join("");

  function show(index){
    range.value = String(index);
    range.setAttribute("aria-valuetext", content.stages[index].name);
    document.querySelector(".method-control").style.setProperty("--fill", `${(index / 3) * 100}%`);
    names.querySelectorAll("button").forEach(button => {
      if (Number(button.dataset.index) === index) button.setAttribute("aria-current", "step");
      else button.removeAttribute("aria-current");
    });
    const apply = () => {
      clause.textContent = content.stages[index].clause;
      note.textContent = NOTES[index](values);
      fragment.innerHTML = builders[index]();
      clause.classList.remove("swap");
      fragment.classList.remove("swap");
    };
    if (reduced) { apply(); return; }
    clause.classList.add("swap");
    fragment.classList.add("swap");
    setTimeout(apply, 200);
  }

  range.addEventListener("input", () => show(Number(range.value)));
  names.addEventListener("click", event => {
    const button = event.target.closest("button[data-index]");
    if (button) show(Number(button.dataset.index));
  });
  show(0);
}
