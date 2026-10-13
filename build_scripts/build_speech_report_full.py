# -*- coding: utf-8 -*-
"""
Full Assembly Script for Speech Processing Comprehensive Report.
Generates: 'd:\\Speech\\PROJECT\\reports\\WildSpoof_Project_Execution_Results_and_Code_Report.docx'
"""

import os
import sys
import docx
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.opc.constants import RELATIONSHIP_TYPE

def add_hyperlink(paragraph, url, text, color="0000EE", underline=True):
    part = paragraph.part
    r_id = part.relate_to(url, RELATIONSHIP_TYPE.HYPERLINK, is_external=True)

    hyperlink = OxmlElement('w:hyperlink')
    hyperlink.set(qn('r:id'), r_id)

    new_run = OxmlElement('w:r')
    rPr = OxmlElement('w:rPr')

    font_elem = OxmlElement('w:rFonts')
    font_elem.set(qn('w:ascii'), 'Times New Roman')
    font_elem.set(qn('w:hAnsi'), 'Times New Roman')
    rPr.append(font_elem)

    size_elem = OxmlElement('w:sz')
    size_elem.set(qn('w:val'), '20')  # 10 pt
    rPr.append(size_elem)

    if color:
        c = OxmlElement('w:color')
        c.set(qn('w:val'), color)
        rPr.append(c)

    if underline:
        u = OxmlElement('w:u')
        u.set(qn('w:val'), 'single')
        rPr.append(u)

    new_run.append(rPr)
    text_elem = OxmlElement('w:t')
    text_elem.text = text
    new_run.append(text_elem)
    hyperlink.append(new_run)
    paragraph._p.append(hyperlink)

def set_cell_borders(cell, top=None, bottom=None, left=None, right=None):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcBorders = tcPr.first_child_found_in('w:tcBorders')
    if tcBorders is None:
        tcBorders = OxmlElement('w:tcBorders')
        tcPr.append(tcBorders)
    
    borders = {'top': top, 'bottom': bottom, 'left': left, 'right': right}
    for edge, val in borders.items():
        if val is not None:
            tag = f'w:{edge}'
            element = tcBorders.find(qn(tag))
            if element is None:
                element = OxmlElement(tag)
                tcBorders.append(element)
            for k, v in val.items():
                element.set(qn(f'w:{k}'), str(v))

def set_cell_shading(cell, color_hex):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd = tcPr.first_child_found_in('w:shd')
    if shd is None:
        shd = OxmlElement('w:shd')
        tcPr.append(shd)
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), color_hex)

def set_cell_margins(cell, top=80, bottom=80, left=110, right=110):
    tc = cell._tc
    tcPr = tc.get_or_add_tcPr()
    tcMar = tcPr.first_child_found_in('w:tcMar')
    if tcMar is None:
        tcMar = OxmlElement('w:tcMar')
        tcPr.append(tcMar)
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)

def add_code_block(doc, title, code_text):
    p_title = doc.add_paragraph()
    p_title.paragraph_format.space_before = Pt(11)
    p_title.paragraph_format.space_after = Pt(2)
    p_title.paragraph_format.keep_with_next = True
    run_t = p_title.add_run(f"Listing: {title}")
    run_t.font.name = 'Consolas'
    run_t.font.size = Pt(10.0)
    run_t.bold = True
    run_t.font.color.rgb = RGBColor(15, 23, 42)

    table = doc.add_table(rows=1, cols=1)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    cell = table.cell(0, 0)
    cell.width = Inches(6.8)

    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), 'F8FAFC')
    tcPr.append(shd)

    borders = OxmlElement('w:tcBorders')
    left = OxmlElement('w:left')
    left.set(qn('w:val'), 'single')
    left.set(qn('w:sz'), '24')  # 3 pt
    left.set(qn('w:color'), '2563EB')
    borders.append(left)
    for b_name in ['top', 'bottom', 'right']:
        b = OxmlElement(f'w:{b_name}')
        b.set(qn('w:val'), 'none')
        borders.append(b)
    tcPr.append(borders)

    set_cell_margins(cell, top=120, bottom=120, left=150, right=150)

    p = cell.paragraphs[0]
    p.paragraph_format.space_before = Pt(3)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.line_spacing = 1.08
    run = p.add_run(code_text.strip())
    run.font.name = 'Consolas'
    run.font.size = Pt(9.0)
    run.font.color.rgb = RGBColor(30, 41, 59)

    p_post = doc.add_paragraph()
    p_post.paragraph_format.space_before = Pt(0)
    p_post.paragraph_format.space_after = Pt(6)
    return table

def add_heading_1(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(18)
    p.paragraph_format.space_after = Pt(5)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(15.0)
    run.bold = True
    run.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_heading_2(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(13)
    p.paragraph_format.space_after = Pt(4)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(13.0)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 58, 138)
    return p

def add_heading_3(doc, text):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(text)
    run.font.name = 'Times New Roman'
    run.font.size = Pt(11.5)
    run.bold = True
    run.font.color.rgb = RGBColor(30, 41, 59)
    return p

def add_body_p(doc, text="", space_after=5):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_before = Pt(0)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.18
    if text:
        run = p.add_run(text)
        run.font.name = 'Times New Roman'
        run.font.size = Pt(11.0)
        run.font.color.rgb = RGBColor(31, 41, 55)
    return p

def add_callout(doc, title, text, border_color="2563EB", bg_color="F8FAFC"):
    tbl = doc.add_table(rows=1, cols=1)
    tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
    c = tbl.cell(0, 0)
    c.width = Inches(6.8)
    set_cell_shading(c, bg_color)
    set_cell_borders(c, left={'val': 'single', 'sz': 24, 'color': border_color})
    set_cell_margins(c, 100, 100, 140, 140)
    p = c.paragraphs[0]
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.line_spacing = 1.15
    r_t = p.add_run(title + "\n")
    r_t.font.name = 'Times New Roman'
    r_t.font.size = Pt(11.0)
    r_t.bold = True
    r_t.font.color.rgb = RGBColor(15, 23, 42)
    r_txt = p.add_run(text)
    r_txt.font.name = 'Times New Roman'
    r_txt.font.size = Pt(10.5)
    r_txt.font.color.rgb = RGBColor(31, 41, 55)
    doc.add_paragraph().paragraph_format.space_after = Pt(4)

def add_image_with_caption(doc, img_path, caption_num, caption_title, caption_desc, width=Inches(6.2)):
    if not os.path.exists(img_path):
        print(f"Warning: Image not found: {img_path}")
        return
    p_img = doc.add_paragraph()
    p_img.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_img.paragraph_format.space_before = Pt(10)
    p_img.paragraph_format.space_after = Pt(4)
    run = p_img.add_run()
    run.add_picture(img_path, width=width)

    p_cap = doc.add_paragraph()
    p_cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_cap.paragraph_format.space_before = Pt(2)
    p_cap.paragraph_format.space_after = Pt(10)
    p_cap.paragraph_format.line_spacing = 1.10

    run1 = p_cap.add_run(f"Figure {caption_num}: {caption_title}. ")
    run1.font.name = 'Times New Roman'
    run1.font.size = Pt(9.5)
    run1.bold = True
    run1.font.color.rgb = RGBColor(15, 23, 42)

    run2 = p_cap.add_run(caption_desc)
    run2.font.name = 'Times New Roman'
    run2.font.size = Pt(9.5)
    run2.font.color.rgb = RGBColor(55, 65, 81)

def add_table_title(doc, table_num, title_text):
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(3)
    p.paragraph_format.keep_with_next = True
    run = p.add_run(f"Table {table_num}: {title_text}")
    run.font.name = 'Times New Roman'
    run.font.size = Pt(10.0)
    run.bold = True
    run.font.color.rgb = RGBColor(15, 23, 42)
    return p

def add_qa_item(doc, q_num, question, answer):
    p_q = doc.add_paragraph()
    p_q.paragraph_format.space_before = Pt(10)
    p_q.paragraph_format.space_after = Pt(2)
    p_q.paragraph_format.keep_with_next = True
    r_q = p_q.add_run(f"Question {q_num}: {question}")
    r_q.font.name = 'Times New Roman'
    r_q.font.size = Pt(11.5)
    r_q.bold = True
    r_q.font.color.rgb = RGBColor(30, 58, 138)

    p_a = doc.add_paragraph()
    p_a.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_a.paragraph_format.space_before = Pt(0)
    p_a.paragraph_format.space_after = Pt(6)
    p_a.paragraph_format.line_spacing = 1.18
    r_lbl = p_a.add_run("Technical Answer: ")
    r_lbl.font.name = 'Times New Roman'
    r_lbl.font.size = Pt(11.0)
    r_lbl.bold = True
    r_lbl.font.color.rgb = RGBColor(15, 23, 42)
    r_a = p_a.add_run(answer)
    r_a.font.name = 'Times New Roman'
    r_a.font.size = Pt(11.0)
    r_a.font.color.rgb = RGBColor(51, 65, 85)

print("Assembly helper script template verified.")
