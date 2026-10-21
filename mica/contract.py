"""/v1/systemone request contract: validation, rows, multi-answer expansion and the response envelope.

``rows_from_request`` turns a request into one independent row per question. ``answer_request`` runs those rows
through any batch probability backend and packages the typed answers. A ``multi`` Choice is answered as one
inclusion Noul per candidate; the count distribution is the Poisson-binomial of the independent inclusion
probabilities.
"""
from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from .native import OPTION_LABELS, typed_result


def _text(value: Any, *, fallback: str | None = None) -> str:
    if value is None and fallback is not None:
        return fallback
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("instructions and criteria descriptions must not be empty")
        return value
    if not isinstance(value, (Mapping, list)) or not value:
        raise ValueError("instructions and criteria must be strings, non-empty objects, or non-empty arrays")
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("instructions or criteria are not valid JSON") from exc


def _contains_image_payload(value: Any) -> bool:
    if isinstance(value, Mapping):
        if str(value.get("modality", "")).lower() == "image" or str(value.get("type", "")).lower() == "image":
            return True
        if value.get("image_url") is not None or value.get("image_bytes") is not None:
            return True
        if isinstance(value.get("images"), list) and value["images"]:
            return True
        return any(_contains_image_payload(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_image_payload(child) for child in value)
    return False


def rows_from_request(request: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Validate a System One request and create independent rows, one per question."""
    if not isinstance(request, Mapping):
        raise ValueError("request must be an object")
    unknown = set(request) - {"model", "state", "questions"}
    if unknown:
        if "images" in unknown:
            raise ValueError("Image inputs are not supported by the local text runtime")
        raise ValueError(f"Unknown request fields: {sorted(unknown)!r}")
    if "state" not in request or request["state"] is None:
        raise ValueError("request.state is required")
    if _contains_image_payload(request["state"]):
        raise ValueError("Image inputs are not supported by the local text runtime")
    questions = request.get("questions")
    if not isinstance(questions, Mapping) or not questions:
        raise ValueError("request.questions must be a non-empty object")

    rows: list[dict[str, Any]] = []
    for question_id, question in questions.items():
        if not isinstance(question_id, str) or not question_id:
            raise ValueError("question IDs must be non-empty strings")
        if not isinstance(question, Mapping):
            raise ValueError(f"question {question_id!r} must be an object")
        question_unknown = set(question) - {"type", "instructions", "criteria", "allow_cannot_grade", "multi", "abstain_threshold"}
        if question_unknown:
            raise ValueError(f"question {question_id!r} has unknown fields: {sorted(question_unknown)!r}")
        kind = str(question.get("type", "")).lower()
        if kind not in {"noul", "choice", "score"}:
            raise ValueError(f"question {question_id!r} has unsupported type {kind!r}")
        allow_cannot_grade = question.get("allow_cannot_grade", False)
        if "allow_cannot_grade" in question:
            if kind != "score" or type(allow_cannot_grade) is not bool:
                raise ValueError("allow_cannot_grade must be a boolean on a Score question")
        if "instructions" not in question:
            raise ValueError(f"question {question_id!r} requires instructions")
        instructions = _text(question["instructions"])
        criteria = question.get("criteria")

        if kind == "noul":
            if criteria is None or (isinstance(criteria, Mapping) and not criteria):
                candidates = [{"id": "false", "text": "No"}, {"id": "true", "text": "Yes"}]
            else:
                if not isinstance(criteria, Mapping) or {str(key).lower() for key in criteria} != {"false", "true"}:
                    raise ValueError(f"Noul question {question_id!r} criteria must contain false and true")
                # The criteria keep the request's order; the readout order is always [false, true].
                by_id = {str(key).lower(): value for key, value in criteria.items()}
                candidates = [
                    {"id": key, "text": _text(value, fallback="No" if key == "false" else "Yes")}
                    for key, value in by_id.items()
                ]
        elif kind == "choice":
            if not isinstance(criteria, Mapping) or not 2 <= len(criteria) <= len(OPTION_LABELS):
                raise ValueError(
                    f"Choice question {question_id!r} requires 2..{len(OPTION_LABELS)} criteria"
                )
            candidates = []
            for candidate_id, description in criteria.items():
                if not isinstance(candidate_id, str) or not candidate_id:
                    raise ValueError(f"Choice question {question_id!r} candidate IDs must be non-empty strings")
                candidates.append({"id": candidate_id, "text": _text(description, fallback=candidate_id)})
        else:
            limit = len(OPTION_LABELS) - int(allow_cannot_grade)
            if not isinstance(criteria, list) or not 2 <= len(criteria) <= limit:
                suffix = " when allow_cannot_grade is true (one label is reserved)" if allow_cannot_grade else ""
                raise ValueError(f"Score question {question_id!r} requires 2..{limit} ordered criteria{suffix}")
            candidates = [
                {"id": str(index), "text": _text(description), "ordinal": index}
                for index, description in enumerate(criteria)
            ]
            if allow_cannot_grade:
                candidates.append({"id": "cannot_grade", "text": "Cannot grade from the supplied state"})
        rows.append(
            {
                "id": question_id,
                "kind": kind,
                "state": copy.deepcopy(request["state"]),
                "question": instructions,
                "candidates": candidates,
                "source": {
                    "schema": "systemone_request",
                    "request_model": request.get("model"),
                },
            }
        )
        if allow_cannot_grade:
            rows[-1]["allow_cannot_grade"] = True
        if "multi" in question:  # every correct candidate: an inclusion probability per candidate
            if kind != "choice" or type(question["multi"]) is not bool:
                raise ValueError("multi must be a boolean on a Choice question")
            if question["multi"]:
                rows[-1]["multi"] = True
        if "abstain_threshold" in question:  # abstain when the calibrated top probability is below this
            t = question["abstain_threshold"]
            if type(t) not in (int, float) or not 0 <= t <= 1:
                raise ValueError("abstain_threshold must be a number in [0, 1]")
            rows[-1]["abstain_threshold"] = float(t)
    return rows


# ---- multi-answer Choice: one inclusion Noul per candidate ----

MULTI_TEMPLATES = {
    'en': '{q}\nOptions: {menu}\nIs this option one of the correct answers? -> {opt}',
    'ko': '{q}\n보기: {menu}\n이 보기가 정답에 포함됩니까? -> {opt}',
    'zh': '{q}\n选项：{menu}\n这个选项是正确答案之一吗？-> {opt}',
    'ja': '{q}\n選択肢：{menu}\nこの選択肢は正解に含まれますか？-> {opt}',
}
INCLUSION_CANDIDATES = [dict(id='true', text='True'), dict(id='false', text='False')]


def inclusion_question(row: Mapping[str, Any], candidate: Mapping[str, Any]) -> str:
    template = MULTI_TEMPLATES.get(row.get('lang'), MULTI_TEMPLATES['en'])
    menu = ' | '.join(c['text'] for c in row['candidates'])
    return template.format(q=row['question'], menu=menu, opt=candidate['text'])


def inclusion_rows(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The inclusion Noul rows of one multi-answer Choice row."""
    out = []
    for c in row['candidates']:
        out.append(dict(id=f"{row['id']}~{c['id']}", kind='noul', state=row['state'], multi_of=row['id'],
                        multi_option=c['id'], question=inclusion_question(row, c),
                        candidates=[dict(x) for x in INCLUSION_CANDIDATES], **({'lang': row['lang']} if 'lang' in row else {})))
    return out


def count_distribution(probabilities: Sequence[float]) -> list[float]:
    dist = [1.]
    for q in probabilities:
        dist = [a * (1 - q) + (dist[i - 1] * q if i else 0.) for i, a in enumerate(dist + [0.])]
    return dist


def assemble(row: Mapping[str, Any], inclusion: Mapping[str, float]) -> dict[str, Any]:
    """inclusion: {candidate id: p(included)} -> the multi-answer result."""
    ids = [c['id'] for c in row['candidates']]
    p = [float(inclusion[i]) for i in ids]
    counts = count_distribution(p)
    return dict(type='choice', multi=True, inclusion=dict(zip(ids, p)), selected=[i for i, v in zip(ids, p) if v >= .5],
                count_distribution=counts, expected_count=sum(k * v for k, v in enumerate(counts)))


# ---- response ----

def response_envelope(
    rows: Sequence[Mapping[str, Any]],
    probabilities: Sequence[Sequence[float]],
    *,
    model: str,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Package typed results without inventing token accounting.

    ``usage`` is omitted unless the caller supplies actual measurements. Token count fields, when supplied, must
    be non-negative integers.
    """
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty local model id")
    if not rows or len(rows) != len(probabilities):
        raise ValueError("Rows and probabilities must have the same non-zero length")
    answers: dict[str, dict[str, Any]] = {}
    for row, distribution in zip(rows, probabilities):
        record_id = row.get("id")
        if not isinstance(record_id, str) or not record_id:
            raise ValueError("Every response row requires a non-empty string id")
        if record_id in answers:
            raise ValueError(f"Duplicate response id {record_id!r}")
        result = typed_result(dict(row), list(distribution))
        if "abstain_threshold" in row:  # the distribution is already temperature-calibrated by the backend
            confidence = max(distribution)
            result = {**result, "confidence_pmax": confidence, "abstain": confidence < row["abstain_threshold"]}
        answers[record_id] = result
    response: dict[str, Any] = {"model": model, "answers": answers}
    if usage is not None:
        if not isinstance(usage, Mapping):
            raise ValueError("usage must be an object")
        for key, value in usage.items():
            if str(key).endswith("_tokens") and (type(value) is not int or value < 0):
                raise ValueError("usage token counts must be non-negative integers")
        response["usage"] = copy.deepcopy(dict(usage))
    return response


def answer_request(
    request: Mapping[str, Any],
    predict: Callable[[list[dict[str, Any]]], Sequence[Sequence[float]]],
    *,
    model: str,
    usage: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a validated request through a batch probability backend.

    ``predict`` receives every row to score (plain questions first, then the inclusion rows of multi-answer
    questions) and returns one distribution per row; Noul distributions are in [false, true] order.
    """
    if not callable(predict):
        raise ValueError("predict must be callable")
    rows = rows_from_request(request)
    multi = [row for row in rows if row.get("multi")]
    plain = [row for row in rows if not row.get("multi")]
    expanded = [r for row in multi for r in inclusion_rows(row)]
    predicted = list(predict(plain + expanded)) if plain or expanded else []
    multi_results = {}
    if expanded:
        inclusion = {}
        for r, dist in zip(expanded, predicted[len(plain):]):
            dist = dist.tolist() if hasattr(dist, "tolist") else list(dist)
            inclusion.setdefault(r["multi_of"], {})[r["multi_option"]] = float(dist[1])  # Noul order is [false, true]
        multi_results = {row["id"]: assemble(row, inclusion[row["id"]]) for row in multi}
        predicted = predicted[:len(plain)]
    rows = plain
    if hasattr(predicted, "tolist"):
        predicted = predicted.tolist()
    if not isinstance(predicted, Sequence):
        raise ValueError("predict must return one probability sequence per question")
    distributions = []
    for distribution in predicted:
        if hasattr(distribution, "tolist"):
            distribution = distribution.tolist()
        if not isinstance(distribution, Sequence):
            raise ValueError("predict must return one probability sequence per question")
        distributions.append(list(distribution))
    response = response_envelope(rows, distributions, model=model, usage=usage) if rows else {"model": model, "answers": {}}
    response["answers"].update(multi_results)
    return response
