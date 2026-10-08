"""从 dishes/ Markdown 菜谱生成可用于 RAG 评测的问答测试集。

生成结果采用 backend.evaluation.EvalDatasetManager 兼容的 JSON 结构：
每条样本包含 question、ground_truth、expected_docs 和 metadata。
脚本不读取图片，也不会凭空补充菜谱没有写出的营养、保存或安全结论。
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DISHES_ROOT = PROJECT_ROOT.parent / "dishes"
OUTPUT_PATH = PROJECT_ROOT / "data" / "eval_samples_dishes.json"

# 正例 108 条，按目录规模分配；另加 12 条文档外问题测试拒答能力。
POSITIVE_TARGETS = {
    "meat_dish": 32,
    "vegetable_dish": 18,
    "staple": 15,
    "aquatic": 8,
    "breakfast": 7,
    "drink": 7,
    "soup": 7,
    "dessert": 6,
    "condiment": 4,
    "semi-finished": 4,
}

TYPE_CYCLE = (
    "ingredients",
    "process",
    "calculation",
    "detail",
    "notes",
    "difficulty",
    "process",
    "ingredients",
    "calculation",
    "detail",
)

UNSUPPORTED_TOPICS = (
    ("热量", re.compile(r"热量|卡路里|千卡")),
    ("保存期限", re.compile(r"保存期限|保质期|保存时间|冷藏|冷冻保存")),
    ("作者", re.compile(r"作者|创作者|发布者")),
    ("起源", re.compile(r"起源|发源地|历史")),
    ("成本", re.compile(r"成本|价格|售价|花费")),
    ("营养成分", re.compile(r"营养成分|蛋白质|脂肪|碳水")),
)


def clean_text(value: str) -> str:
    """移除 Markdown 图片/链接和多余格式，保留菜谱正文。"""
    value = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", value)
    value = re.sub(r"[*_`]+", "", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" \t-：:")


def lines_without_images(value: str) -> list[str]:
    lines: list[str] = []
    for raw in value.splitlines():
        line = clean_text(raw)
        if not line or line.startswith("!"):
            continue
        lines.append(line)
    return lines


def section(text: str, heading: str) -> str:
    """提取一个二级标题到下一个二级标题之间的内容。"""
    match = re.search(
        rf"^##\s*{re.escape(heading)}\s*$([\s\S]*?)(?=^##\s+|\Z)",
        text,
        flags=re.MULTILINE,
    )
    return match.group(1).strip() if match else ""


def first_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)\s*$", text, flags=re.MULTILINE)
    title = clean_text(match.group(1)) if match else fallback
    return re.sub(r"的做法$", "", title).strip() or fallback


def intro_text(text: str) -> str:
    match = re.search(r"^#\s+.+?$([\s\S]*?)(?=^##\s+|\Z)", text, flags=re.MULTILINE)
    if not match:
        return ""
    intro = clean_text(match.group(1))
    # 难度单独作为 metadata，避免在描述答案中重复一次。
    intro = re.sub(r"预估烹饪难度\s*[:：]\s*[^。；\n]+[。；]?", "", intro)
    return intro.strip()


def bullet_lines(value: str) -> list[str]:
    result: list[str] = []
    for line in lines_without_images(value):
        # 保留有序步骤中的文字，但去掉 Markdown 项目符号。
        line = re.sub(r"^\d+[.)、]\s*", "", line)
        if line.startswith(("-", "+", "•", "*")):
            line = line.lstrip("-+*• ")
        if line:
            result.append(line)
    return result


def compact_lines(lines: Iterable[str], limit: int = 1800) -> str:
    """以分号连接正文，避免单条 ground_truth 无限膨胀。"""
    parts: list[str] = []
    used = 0
    for line in lines:
        line = clean_text(line)
        if not line:
            continue
        addition = ("；" if parts else "") + line
        if used + len(addition) > limit:
            break
        parts.append(line)
        used += len(addition)
    return "；".join(parts)


def parse_recipe(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    relative = path.relative_to(PROJECT_ROOT.parent).as_posix()
    category = path.relative_to(DISHES_ROOT).parts[0]
    title = first_title(text, path.stem)
    intro = intro_text(text)
    ingredients = bullet_lines(section(text, "必备原料和工具"))
    calculation = bullet_lines(section(text, "计算"))
    operations = bullet_lines(section(text, "操作"))
    notes = bullet_lines(section(text, "附加内容"))
    difficulty_match = re.search(r"预估烹饪难度\s*[:：]\s*([^\n]+)", text)
    difficulty = clean_text(difficulty_match.group(1)) if difficulty_match else ""

    # 选取一条包含火候、时间或关键动作的原文，作为精确细节题答案。
    detail = next(
        (
            line
            for line in operations + notes
            if re.search(r"大火|中火|小火|分钟|小时|℃|度|腌制|浸泡|翻炒|蒸|烤|煮|不可|不能|切忌|关键", line)
        ),
        operations[0] if operations else (notes[0] if notes else ""),
    )
    return {
        "path": relative,
        "category": category,
        "title": title,
        "raw": text,
        "intro": intro,
        "ingredients": ingredients,
        "calculation": calculation,
        "operations": operations,
        "notes": notes,
        "difficulty": difficulty,
        "detail": detail,
    }


def base_metadata(recipe: dict, question_type: str, unanswerable: bool = False) -> dict:
    return {
        "category": recipe["category"],
        "dish": recipe["title"],
        "difficulty": recipe["difficulty"],
        "question_type": question_type,
        "source_type": "markdown",
        "unanswerable": unanswerable,
    }


def make_positive_candidates(recipe: dict) -> list[dict]:
    title = recipe["title"]
    path = recipe["path"]
    candidates: list[dict] = []

    if recipe["ingredients"]:
        answer = compact_lines(recipe["ingredients"], 1100)
        candidates.append({
            "question": f"制作{title}需要哪些主要原料和工具？",
            "ground_truth": answer,
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "ingredients"),
        })

    if len(recipe["calculation"]) >= 2:
        answer = compact_lines(recipe["calculation"], 1300)
        candidates.append({
            "question": f"{title}的配料用量或份量如何计算？",
            "ground_truth": answer,
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "calculation"),
        })

    if len(recipe["operations"]) >= 3:
        answer = compact_lines(recipe["operations"], 1800)
        candidates.append({
            "question": f"请按顺序概述{title}的主要制作步骤。",
            "ground_truth": answer,
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "process"),
        })

    if recipe["detail"]:
        candidates.append({
            "question": f"制作{title}时，文档强调的一个关键火候或操作细节是什么？",
            "ground_truth": recipe["detail"],
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "detail"),
        })

    if recipe["notes"]:
        answer = compact_lines(recipe["notes"], 1300)
        candidates.append({
            "question": f"制作{title}有哪些附加注意事项？",
            "ground_truth": answer,
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "notes"),
        })

    if recipe["difficulty"] or recipe["intro"]:
        prefix = f"预估烹饪难度为{recipe['difficulty']}。" if recipe["difficulty"] else ""
        answer = (prefix + recipe["intro"]).strip()
        candidates.append({
            "question": f"{title}的烹饪难度如何？文档怎样描述这道菜？",
            "ground_truth": answer[:1000],
            "expected_docs": [path],
            "metadata": base_metadata(recipe, "difficulty"),
        })

    return [c for c in candidates if c["ground_truth"]]


def make_negative_candidate(recipe: dict) -> dict | None:
    title = recipe["title"]
    for topic, pattern in UNSUPPORTED_TOPICS:
        if not pattern.search(recipe["raw"]):
            return {
                "question": f"{title}的{topic}是多少？",
                "ground_truth": f"{title}的菜谱文档未提供{topic}信息，无法仅根据该文档确定。",
                "expected_docs": [recipe["path"]],
                "metadata": base_metadata(recipe, "unanswerable", True),
            }
    return None


def choose_positive_samples(recipes_by_category: dict[str, list[dict]]) -> list[dict]:
    selected: list[dict] = []
    for category, target in POSITIVE_TARGETS.items():
        recipes = recipes_by_category[category]
        candidates = [c for recipe in recipes for c in make_positive_candidates(recipe)]
        by_type: dict[str, list[dict]] = defaultdict(list)
        for candidate in candidates:
            by_type[candidate["metadata"]["question_type"]].append(candidate)

        used_questions: set[str] = set()
        used_sources: Counter[str] = Counter()
        for index in range(target):
            preferred = TYPE_CYCLE[index % len(TYPE_CYCLE)]
            pool = by_type.get(preferred, [])
            if not pool:
                pool = [c for c in candidates if c["question"] not in used_questions]
            # 先覆盖更多菜谱，避免一个长菜谱贡献过多题目。
            available = [c for c in pool if c["question"] not in used_questions]
            if not available:
                break
            available.sort(key=lambda c: (used_sources[c["expected_docs"][0]], c["expected_docs"][0]))
            chosen = available[0]
            selected.append(chosen)
            used_questions.add(chosen["question"])
            used_sources[chosen["expected_docs"][0]] += 1
    return selected


def choose_negative_samples(recipes: list[dict], count: int = 12) -> list[dict]:
    selected: list[dict] = []
    seen_categories: set[str] = set()
    # 第一轮覆盖不同目录，第二轮再补足数量。
    for recipe in recipes:
        if len(selected) >= count:
            break
        if recipe["category"] in seen_categories:
            continue
        candidate = make_negative_candidate(recipe)
        if candidate:
            selected.append(candidate)
            seen_categories.add(recipe["category"])
    for recipe in recipes:
        if len(selected) >= count:
            break
        candidate = make_negative_candidate(recipe)
        if candidate and candidate["question"] not in {s["question"] for s in selected}:
            selected.append(candidate)
    return selected


def main() -> None:
    paths = sorted(
        p for p in DISHES_ROOT.rglob("*.md") if "template" not in p.relative_to(DISHES_ROOT).parts
    )
    recipes = [parse_recipe(path) for path in paths]
    recipes_by_category: dict[str, list[dict]] = defaultdict(list)
    for recipe in recipes:
        recipes_by_category[recipe["category"]].append(recipe)

    positives = choose_positive_samples(recipes_by_category)
    negatives = choose_negative_samples(recipes, 12)
    samples = positives + negatives

    # 生成后做基本一致性校验。
    assert len(samples) == 120, f"期望 120 条样本，实际 {len(samples)} 条"
    assert len({s["question"] for s in samples}) == len(samples), "存在重复问题"
    for sample in samples:
        assert sample["question"] and sample["ground_truth"]
        for source in sample["expected_docs"]:
            assert (PROJECT_ROOT.parent / source).exists(), f"来源不存在: {source}"

    category_counts = Counter(s["metadata"]["category"] for s in samples)
    type_counts = Counter(s["metadata"]["question_type"] for s in samples)
    payload = {
        "export_time": datetime.now(timezone.utc).isoformat(),
        "description": "基于 dishes/ Markdown 菜谱生成的 SuperMew RAG 评测数据集；图片和 template 示例不纳入文本评测。",
        "source_root": "dishes/",
        "source_markdown_files": len(recipes),
        "total_samples": len(samples),
        "positive_samples": len(positives),
        "unanswerable_samples": len(negatives),
        "category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(sorted(type_counts.items())),
        "samples": samples,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(OUTPUT_PATH),
        "source_markdown_files": len(recipes),
        "total_samples": len(samples),
        "positive_samples": len(positives),
        "unanswerable_samples": len(negatives),
        "category_counts": dict(sorted(category_counts.items())),
        "question_type_counts": dict(sorted(type_counts.items())),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
