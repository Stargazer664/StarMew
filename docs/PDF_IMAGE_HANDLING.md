# PDF 图片处理说明

## 当前处理方式

### 图片处理：忽略

**当前使用**：`PyPDFLoader`（LangChain）

```python
# document_loader.py:149-155
if file_lower.endswith(".pdf"):
    doc_type = "PDF"
    loader = PyPDFLoader(file_path)

# 加载结果
raw_docs = loader.load()
# doc.page_content = 纯文本（图片被忽略）
```

### 处理流程

```
PDF 文件
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│                      PyPDFLoader                           │
│  (仅提取文本内容，图片/图表/表格 → 丢弃)                    │
└─────────────────────────────────────────────────────────────┘
    │
    ├──► doc.page_content ──► 纯文本字符串
    │
    └──► 图片/表格/公式 ──► 丢弃（不处理）
           │
           ▼
    文本送入分块流程
```

---

## 图片被忽略的原因

| 组件 | 说明 |
|------|------|
| **PyPDFLoader** | 使用 `pypdf` 库，只读取文本层 |
| **分块逻辑** | `_split_page_to_three_levels` 只处理 `page_content` |
| **存储** | 只存储 `text` 字段，图片数据丢失 |

---

## 图片处理方案

### 方案一：保留图片（推荐）

如果需要处理 PDF 中的图片，可以替换为 `PyMuPDF` (FITZ) 或使用 `marker` 等专用工具：

```python
# 方案 A: PyMuPDF - 提取图片 + 文本
import fitz  # pip install pymupdf

def load_pdf_with_images(file_path):
    doc = fitz.open(file_path)
    chunks = []

    for page_num, page in enumerate(doc):
        # 提取文本
        text = page.get_text()

        # 提取图片
        images = []
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            images.append({
                "page": page_num,
                "index": img_index,
                "image_bytes": base_image["image"],
                "image_ext": base_image["ext"],
            })

        chunks.append({
            "text": text,
            "images": images,
            "page_number": page_num,
        })

    return chunks
```

### 方案二：OCR + 图片描述（完整方案）

使用 `marker` 或 `unstructured` 的高级 PDF 处理器：

```bash
pip install marker-pipeline  # 或 unstructured[pdf]
```

```python
from marker.convert import convert_single_pdf

def load_pdf_with_ocr(file_path):
    """使用 marker 进行 PDF 渲染 + OCR"""
    rendered = convert_single_pdf(
        file_path,
        doc_builder=DocBuilder(
            table_as_text=True,
            formula_as_text=True,
            imgs_to_pdf=False,  # 不转回 PDF
        )
    )

    # rendered.blocks 包含：
    # - TextBlock: 文本块
    # - ImageBlock: 图片块（带 OCR 描述）
    # - TableBlock: 表格块

    for block in rendered.blocks:
        if isinstance(block, ImageBlock):
            # block.image  # 图片数据
            # block.text   # 图片描述/OCR 结果
            pass

    return rendered
```

---

## 多模态 RAG 方案

如果需要让 LLM "看懂" 图片，需要多模态 RAG：

```
图片 ──► 多模态 Embedding ──► 向量存储
  │           (如: CLIP, JinaCLIP)
  │                              │
  │                              ▼
  │                         检索图片
  │                              │
  └──────────────────────────────┘
            │
            ▼
      LLM 理解图片内容
```

### 实现示例

```python
from transformers import CLIPProcessor, CLIPModel

class MultimodalRAG:
    def __init__(self):
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def embed_image(self, image_bytes):
        from PIL import Image
        import io
        image = Image.open(io.BytesIO(image_bytes))
        inputs = self.clip_processor(images=image, return_tensors="pt")
        return self.clip_model.get_image_features(**inputs)

    def embed_text(self, text):
        inputs = self.clip_processor(text=text, return_tensors="pt", padding=True)
        return self.clip_model.get_text_features(**inputs)

    def search(self, query, top_k=5):
        query_emb = self.embed_text(query)
        # 与图片向量匹配
        # ...
```

---

## 表格处理

当前方案也没有特殊处理表格，表格内容被作为纯文本处理：

```
┌─────────┬─────────┐
│ 表格    │ 内容    │  ──► "表格内容作为纯文本"
└─────────┴─────────┘

问题：结构丢失，检索效果差
```

### 表格增强方案

```python
from marker.converters.pdf import PdfConverter
from marker.models import create_model_from_config
from marker.converters.config import PdfConverterConfig

def load_pdf_with_tables(file_path):
    converter = PdfConverter(
        config=PdfConverterConfig(
            table_as_text=True,  # 表格转文本（结构化）
        )
    )
    rendered = converter(file_path)

    for block in rendered.blocks:
        if isinstance(block, TableBlock):
            # block.to_csv()  # 表格转 CSV
            # block.to_markdown()  # 表格转 Markdown
            pass
```

---

## 当前项目扩展建议

如果需要在 StarMew 中添加图片处理能力：

### 1. 修改 DocumentLoader

```python
# document_loader.py 新增方法

def load_pdf_advanced(self, file_path: str, filename: str, extract_images: bool = False):
    """高级 PDF 加载

    Args:
        extract_images: 是否提取图片
    """
    from langchain_community.document_loaders import UnstructuredPDFLoader

    # 方式一：Unstructured（支持图片）
    loader = UnstructuredPDFLoader(
        file_path,
        strategy="elements",
        extract_images=True
    )

    # 方式二：PyMuPDF
    # loader = PyMuPDFLoader(file_path)  # 需要安装

    docs = loader.load()

    for doc in docs:
        # doc.metadata["image_path"]  # 图片路径
        # doc.page_content  # 包含图片描述
        pass

    return docs
```

### 2. 图片存储

```python
# 新增图片存储表
class PDFImage(Base):
    __tablename__ = "pdf_images"

    id = Column(Integer, primary_key=True)
    chunk_id = Column(String(512), ForeignKey("parent_chunks.chunk_id"))
    page_number = Column(Integer)
    image_data = Column(LargeBinary)  # 或存储路径
    ocr_text = Column(Text)  # 图片 OCR 结果
    embedding = Column(Vector(1024))  # 多模态向量
```

---

## 总结

| 当前状态 | 说明 |
|----------|------|
| **图片处理** | ❌ 忽略（PyPDFLoader 只提取文本） |
| **表格处理** | ⚠️ 作为纯文本（结构丢失） |
| **公式处理** | ⚠️ 可能丢失或乱码 |

### 如果需要图片处理

| 方案 | 复杂度 | 效果 |
|------|--------|------|
| PyMuPDF 提取 | 低 | 提取图片，需额外 OCR |
| marker 工具 | 中 | 端到端 OCR + 表格 + 公式 |
| 多模态 Embedding | 高 | 语义理解图片 |
| GPT-4V / Claude Vision | 高 | 理解图片内容 |

---

*文档更新时间：2026-05-14*