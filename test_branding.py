from html.parser import HTMLParser
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent


class FrontendParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.tags = []
        self.attributes = []
        self.text = []

    def handle_starttag(self, tag, attrs):
        self.tags.append(tag)
        self.attributes.append(dict(attrs))

    def handle_data(self, data):
        self.text.append(data)


def parse_frontend():
    parser = FrontendParser()
    parser.feed((PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8"))
    return parser


def test_visible_branding_uses_xingmiao_everywhere():
    parser = parse_frontend()
    visible_text = "".join(parser.text)
    title = next((text.strip() for text in parser.text if text.strip()), "")

    assert "星喵" in visible_text
    assert "StarMew · 星喵助手" in title
    assert "喵喵" not in visible_text
    assert "🐱" not in visible_text


def test_frontend_uses_one_inline_svg_icon_system():
    parser = parse_frontend()
    attributes = parser.attributes

    assert "i" not in parser.tags
    assert not any("font-awesome" in attrs.get("href", "") for attrs in attributes)
    assert any(
        attrs.get("class") == "icon-sprite"
        for attrs in attributes
        if attrs.get("class")
    )
    assert any(
        attrs.get("id") == "icon-brand"
        for attrs in attributes
    )
    assert sum(1 for tag in parser.tags if tag == "svg") >= 10


def test_agent_prompt_introduces_xingmiao_instead_of_old_name():
    agent_source = (PROJECT_ROOT / "backend" / "agent.py").read_text(encoding="utf-8")

    assert "星喵" in agent_source
    assert "你是一只可爱的猫咪助手" not in agent_source


def test_product_branding_uses_starmew_in_user_facing_surfaces():
    frontend = (PROJECT_ROOT / "frontend" / "index.html").read_text(encoding="utf-8")
    readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    evaluation = (PROJECT_ROOT / "backend" / "evaluation.py").read_text(encoding="utf-8")

    assert "登录 StarMew" in frontend
    assert "注册 StarMew" in frontend
    assert "SuperMew" not in frontend
    assert "# StarMew 项目说明" in readme
    assert "StarMew RAG 评测工具" in evaluation


def test_project_defaults_use_starmew_identifier():
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    cache = (PROJECT_ROOT / "backend" / "cache.py").read_text(encoding="utf-8")
    langsmith = (PROJECT_ROOT / "backend" / "langsmith_config.py").read_text(encoding="utf-8")

    assert 'name = "StarMew"' in pyproject
    assert '"starmew"' in cache
    assert '"starmew-dev"' in langsmith


def test_old_product_brand_is_absent_from_project_branding_surfaces():
    branding_files = [
        "README.md",
        ".env.example",
        "pyproject.toml",
        "docker-compose.yml",
        "run.py",
        "uv.lock",
        "frontend/index.html",
        "backend/agent.py",
        "backend/cache.py",
        "backend/evaluation.py",
        "backend/langsmith_config.py",
        "docs/CROSS_THREAD_RAG.md",
        "docs/MCP_INTEGRATION.md",
        "docs/MEMORY_SYSTEM.md",
        "docs/PDF_IMAGE_HANDLING.md",
        "docs/RESUME_RAG_AGENT.md",
        "docs/SSE_STREAMING.md",
        "docs/TECH_STACK.md",
    ]

    for relative_path in branding_files:
        content = (PROJECT_ROOT / relative_path).read_text(encoding="utf-8")
        assert "SuperMew" not in content
        assert "supermew" not in content
