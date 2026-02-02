# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: rag4pdf
#     language: python
#     name: rag4pdf
# ---

# %%
import os
import sys

# poppler/tesseract(homebrew) 
os.environ["PATH"] = "/opt/homebrew/bin:" + os.environ["PATH"]

# %%
book_name = "数据仓库工具箱维度建模权威指南（第3版）"
chapter = "2"
file = f"books/{book_name}/{chapter}.pdf"

# %%
from pathlib import Path
path = Path(f"outputs/{book_name}/imgs")
path.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 解析 PDF

# %%
from unstructured.partition.auto import partition
loader = partition(
        filename=file,
        strategy="hi_res",  # "fast" or "accurate"
        infer_table_structure=True,
        languages=["chi_sim", "eng"],
        ocr_engine="paddleocr",
        )

doc_local = []
for doc in loader:
    doc_local.append(doc)

print(f"Loaded {len(doc_local)} documents.")
print(doc_local[1])  # Print the first 500 characters of the first document


# %% [markdown]
# ## 提取图片

# %% [markdown]
# ### 提取页面图片

# %%
import fitz

doc = fitz.open(file)  #打开pdf 
print ("number of pages: %i" % doc.page_count)  #获取页码数
print(doc.metadata)  # 获取pdf信息


# %% [markdown]
#
# ### 根据 Unstructured 提供的插图坐标进行截图

# %%

def get_image(element, page):
    
    coord_system = element.metadata.coordinates.system
    u_page_width = coord_system.width
    u_page_height = coord_system.height
    points = element.metadata.coordinates.points
    # 转换为 (x0, y0, x1, y1) 格式: 左上角和右下角
    # points[0] 是左上, points[2] 是右下
    u_x0, u_y0 = points[0]
    u_x1, u_y1 = points[2]

    # --- C. 计算“相对位置” (0.0 ~ 1.0) ---
    # 例如：图片左边框在页面 10% 的位置
    rel_x0 = u_x0 / u_page_width
    rel_y0 = u_y0 / u_page_height
    rel_x1 = u_x1 / u_page_width
    rel_y1 = u_y1 / u_page_height
    
    # --- D. 映射到 Fitz 的实际页面尺寸 ---
    # fitz 的 page.rect.width 才是我们裁剪时真实的参照系
    f_page_width = page.rect.width
    f_page_height = page.rect.height
    
    # 考虑页面可能有 CropBox 偏移 (page.rect.x0 不一定是 0)
    offset_x = page.rect.x0
    offset_y = page.rect.y0
    
    # 计算最终坐标：(相对位置 * 实际宽度) + 起始偏移量
    final_x0 = (rel_x0 * f_page_width) + offset_x
    final_y0 = (rel_y0 * f_page_height) + offset_y
    final_x1 = (rel_x1 * f_page_width) + offset_x
    final_y1 = (rel_y1 * f_page_height) + offset_y

    # --- C. 使用 fitz 进行裁剪 ---
    # 创建裁剪矩形
    fix_x = 8
    fix_y = 8
    rect = fitz.Rect(final_x0-fix_x, final_y0-fix_y-1, final_x1+fix_x, final_y1+fix_y+18)
    rect = rect & page.rect
    if rect.width < 1 or rect.height < 1:
        print(f"跳过无效图片区域: {rect}")
        print(page.rect)
        return False
    zoom = 2  # 2倍缩放 (相当于 144 dpi)
    mat = fitz.Matrix(zoom, zoom)
    
    pix = page.get_pixmap(matrix=mat, clip=rect)

    # 二次检查：防止像素数据生成失败
    if pix.width < 1 or pix.height < 1:
        print(f"跳过无效图片区域2: {rect}")
        return False

    return pix

"""
image_count = 0
image_map = []
for el in doc_local:
    cat = el.category
    pn = el.metadata.page_number
    text =el.text
    if cat in ["Image", "Table"]:
        image_count = image_count + 1
        page = doc[pn-1]

        pix = get_image(el, page)

        img_path = os.path.join(f"outputs/{book_name}/imgs", f"chapter{chapter}-page{pn}_img{image_count}.png")
        pix.save(img_path)
        image_map.append(img_path)
"""

# %% [markdown]
# ## 保存 Markdown

# %%
import re
from unstructured.cleaners.core import clean_bullets

TITLE_PATTERN = re.compile(r'^(\d+(?:\.\d+)*)\s+(.*)$')
HEADER_CHAPTER_PATTERN = re.compile(r'^第(\d+)章\s*(.*?)\s*(\d+)$')
PREV_ELEMENT_IS_LIST = False

md_lines = []
uncategorized_text = []

el = doc_local[1]
image_count = 0

for el in doc_local:
    cat = el.category
    pn = el.metadata.page_number
    text = el.text
    if cat == "Title":
        level = text.count('.') + 1
        if el.metadata.category_depth:
            print(el.metadata.category_depth, text)
        elif level>1:
            md_lines.append("#"*level +f"{text}\n")
        else:
            md_lines.append(f"# {text}\n")
    elif cat == "Header":
        # 页眉或页脚
        # md_lines.append(f"## {text}\n")
        print(f"{pn}, {text}")
    elif cat == "XTable":
        print(el.metadata["text_as_html"])
        if "text_as_html" in el.metadata:
            from html2text import html2text
            md_lines.append(html2text(el.metadata["text_as_html"]) + "\n")
            print(1)
        else:
            md_lines.append(text + "\n")    
            print(2)

    elif cat in ["Image", "Table"]:
        image_count += 1
        page = doc[pn-1]

        pix = get_image(el, page)
        
        if pix:
            img_path = os.path.join("imgs", f"chapter{chapter}-page{pn}_img{image_count}.png")
            pix.save(f"outputs/{book_name}/{img_path}")
            md_lines.append(f"![Image](./{img_path})\n")
    elif cat == "ListItem":
        # 去除 bullets 
        md_lines.append("- " + clean_bullets(text) + "\n")
        PREV_ELEMENT_IS_LIST = True
    elif cat == "NarrativeText":
        if PREV_ELEMENT_IS_LIST:
            md_lines.append("\n")
            PREV_ELEMENT_IS_LIST = False
        md_lines.append(text + "\n")
    else:
        if TITLE_PATTERN.match(text):
            level = text.count('.') + 1
            if el.metadata.category_depth:
                print(el.metadata.category_depth, text)
            elif level>1:
                md_lines.append("#"*level +f"{text}\n")
            else:
                md_lines.append(f"# {text}\n")
        elif HEADER_CHAPTER_PATTERN.match(text):
            print(f"Header: {text}")
        else:
            uncategorized_text.append(el)

if len(uncategorized_text):
    for el in uncategorized_text:
        print(text)

output_md = f"outputs/{book_name}/{chapter}.md"
with open(output_md, "w", encoding="utf-8") as f:
    f.write("".join(md_lines))

# %%
