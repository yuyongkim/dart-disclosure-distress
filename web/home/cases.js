// The case lookup: the headline is an input; typing a firm name shows that firm's record from the paper.
import { escapeHtml } from "./format.js";
import { sequenceList, scoreFrame } from "./case-view.js";

function find(cases, query){
  const text = query.trim().replace(/\s+/g, "");
  if (!text) return null;
  return cases.find(record => record.firm.replace(/\s+/g, "").includes(text)) || null;
}

function answerHtml(record, labels){
  if (!record) return null;
  return `${scoreFrame(record, labels)}
    <h3>기준일까지 모형이 본 공시</h3>${record.visible.length ? sequenceList(record.visible) : "<p>사례 기록에 기준일 전 주요사항보고서가 없습니다. 점수가 상위 10% 밖으로 나와 모형이 놓친 사례입니다.</p>"}
    ${record.after.length ? `<h3>기준일 뒤, 결과까지</h3>${sequenceList(record.after, "after")}` : ""}`;
}

function missHtml(cases){
  return `<p>논문 사례 ${cases.length}곳 가운데 이 이름과 맞는 기업이 없습니다. 사례: ${cases.map(record => escapeHtml(record.firm)).join(", ")}.</p>`;
}

export function mountCases({ cases, content }){
  const input = document.getElementById("caseInput");
  const answer = document.getElementById("caseAnswer");
  const examples = document.getElementById("caseExamples");
  const labels = content.outcome_labels;

  const draw = () => { answer.innerHTML = answerHtml(find(cases, input.value), labels) || missHtml(cases); };

  examples.innerHTML = content.case_examples
    .map(name => `<button type="button" data-name="${escapeHtml(name)}">${escapeHtml(name)}</button>`).join("");
  examples.addEventListener("click", event => {
    const button = event.target.closest("button[data-name]");
    if (!button) return;
    input.value = button.dataset.name;
    draw();
  });
  input.addEventListener("input", draw);
  draw();
}
