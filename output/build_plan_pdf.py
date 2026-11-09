"""Build the readable PDF and standalone checklist from master_plan.md."""
from pathlib import Path
import re
from html import escape

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    BaseDocTemplate, PageTemplate, Frame, Paragraph, Spacer, PageBreak,
    LongTable, TableStyle, Preformatted, KeepTogether,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
import fitz

ROOT = Path(__file__).resolve().parent
PDF_DIR = ROOT / 'pdf'
QA_DIR = ROOT.parent / 'tmp' / 'pdfs'
PDF_DIR.mkdir(parents=True, exist_ok=True)
QA_DIR.mkdir(parents=True, exist_ok=True)

for name, file in [('Plan','calibri.ttf'), ('PlanBold','calibrib.ttf'),
                   ('PlanItalic','calibrii.ttf'), ('PlanMono','consola.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
pdfmetrics.registerFontFamily('Plan', normal='Plan', bold='PlanBold', italic='PlanItalic', boldItalic='PlanBold')

NAVY = colors.HexColor('#17324d')
TEAL = colors.HexColor('#087f8c')
INK = colors.HexColor('#263344')
MUTED = colors.HexColor('#607080')
PALE = colors.HexColor('#f0f5f8')
PAGE_W, PAGE_H = 595.276, 841.89
LEFT, RIGHT, TOP, BOTTOM = 48, 48, 54, 45
WIDTH = PAGE_W - LEFT - RIGHT
styles = getSampleStyleSheet()
styles.add(ParagraphStyle(name='BodyPlan',fontName='Plan',fontSize=9.9,leading=13.6,textColor=INK,spaceAfter=6.5,allowWidows=0,allowOrphans=0))
styles.add(ParagraphStyle(name='H2Plan',fontName='PlanBold',fontSize=19,leading=23,textColor=NAVY,spaceAfter=14,keepWithNext=True))
styles.add(ParagraphStyle(name='H3Plan',fontName='PlanBold',fontSize=12,leading=16,textColor=TEAL,spaceBefore=10,spaceAfter=7,keepWithNext=True))
styles.add(ParagraphStyle(name='BulletPlan',parent=styles['BodyPlan'],leftIndent=12,firstLineIndent=-9,spaceAfter=5))
styles.add(ParagraphStyle(name='TablePlan',fontName='Plan',fontSize=8.4,leading=11.1,textColor=INK,spaceAfter=0,wordWrap='LTR'))
styles.add(ParagraphStyle(name='TableHeadPlan',parent=styles['TablePlan'],fontName='PlanBold',textColor=colors.white))
styles.add(ParagraphStyle(name='CodePlan',fontName='PlanMono',fontSize=8,leading=10.7,textColor=INK,backColor=PALE,borderPadding=8,spaceAfter=9))

md = MarkdownIt('commonmark').enable('table')

def inline(tokens):
    result=[]
    for t in tokens or []:
        if t.type=='text': result.append(escape(t.content).replace('[ ]','[  ]'))
        elif t.type=='code_inline': result.append('<font name="PlanMono" size="8.5">'+escape(t.content)+'</font>')
        elif t.type=='strong_open': result.append('<b>')
        elif t.type=='strong_close': result.append('</b>')
        elif t.type=='em_open': result.append('<i>')
        elif t.type=='em_close': result.append('</i>')
        elif t.type=='link_open': result.append('<a href="'+escape(t.attrGet('href'),quote=True)+'" color="#087f8c">')
        elif t.type=='link_close': result.append('</a>')
        elif t.type in ('softbreak','hardbreak'): result.append(' ')
    return ''.join(result)

class PlanDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename),pagesize=(PAGE_W,PAGE_H),leftMargin=LEFT,rightMargin=RIGHT,topMargin=TOP,bottomMargin=BOTTOM,title='Generative AI Assignment 2 - Complete Master Plan',author='Prepared for the assignment workspace')
        frame=Frame(LEFT,BOTTOM,WIDTH,PAGE_H-TOP-BOTTOM,id='body',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='plan',frames=frame,onPage=self.page_chrome))
    def page_chrome(self, canvas, doc):
        if doc.page==1: return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d9e3e9'))
        canvas.line(LEFT,PAGE_H-34,PAGE_W-RIGHT,PAGE_H-34)
        canvas.setFont('Plan',8)
        canvas.setFillColor(MUTED)
        canvas.drawString(LEFT,PAGE_H-25,'GEN AI ASSIGNMENT 2  /  IMPLEMENTATION MASTER PLAN')
        canvas.drawString(LEFT,25,'Revised 9 November 2026  |  Colab research / local application')
        canvas.drawRightString(PAGE_W-RIGHT,25,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='H2Plan' and flowable.getPlainText()!='Contents':
            title=flowable.getPlainText()
            key='section_'+re.sub(r'[^a-zA-Z0-9]+','_',title)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0,False)
            self.notify('TOCEntry',(0,title,self.page,key))

def table_block(tokens):
    rows=[]; row=[]; header=True; inhead=False
    for t in tokens:
        if t.type=='thead_open': inhead=True
        elif t.type=='thead_close': inhead=False
        elif t.type=='tr_open': row=[]
        elif t.type=='inline': row.append(Paragraph(inline(t.children),styles['TableHeadPlan' if inhead else 'TablePlan']))
        elif t.type=='tr_close': rows.append(row)
    n=len(rows[0])
    first=rows[0][0].getPlainText()
    if n==3 and first=='Code': widths=[0.075*WIDTH,0.52*WIDTH,0.405*WIDTH]
    elif n==3 and first=='Phase': widths=[0.09*WIDTH,0.65*WIDTH,0.26*WIDTH]
    elif n==3: widths=[0.22*WIDTH,0.37*WIDTH,0.41*WIDTH]
    elif n==4: widths=[0.22*WIDTH,0.29*WIDTH,0.21*WIDTH,0.28*WIDTH]
    elif n==2: widths=[0.33*WIDTH,0.67*WIDTH]
    else: widths=[WIDTH/n]*n
    tab=LongTable(rows,colWidths=widths,repeatRows=1,hAlign='LEFT',splitByRow=1)
    tab.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),
        ('LINEBELOW',(0,0),(-1,0),0.6,TEAL),
        ('LINEBELOW',(0,1),(-1,-1),0.25,colors.HexColor('#dce5eb')),
    ]))
    return tab

def body_story(text):
    tokens=md.parse(text); story=[]; i=0; lists=[]; current_section=''
    while i<len(tokens):
        t=tokens[i]
        if t.type=='heading_open':
            level=int(t.tag[1:]); title=tokens[i+1]
            if level==1: i+=3; continue
            if level==2:
                current_section=title.content
                if story: story.append(PageBreak())
                story.append(Paragraph(inline(title.children),styles['H2Plan']))
            else:
                if current_section.startswith('Phase 8') and title.content=='Tests and measurements':
                    story.append(PageBreak())
                    story.append(Paragraph('Phase 8 - Validation and measurements',styles['H3Plan']))
                story.append(Paragraph(inline(title.children),styles['H3Plan']))
            i+=3; continue
        if t.type in ('bullet_list_open','ordered_list_open'):
            lists.append({'ordered':t.type=='ordered_list_open','n':int(t.attrGet('start') or 1)})
        elif t.type in ('bullet_list_close','ordered_list_close'): lists.pop()
        elif t.type=='paragraph_open':
            content=inline(tokens[i+1].children)
            if lists:
                ctx=lists[-1]
                prefix=(str(ctx['n'])+'. ') if ctx['ordered'] else '- '
                ctx['n']+=1
                story.append(Paragraph(prefix+content,styles['BulletPlan']))
            else: story.append(Paragraph(content,styles['BodyPlan']))
            i+=3; continue
        elif t.type=='fence':
            # Wrap only very long source lines at whitespace for print.
            import textwrap
            lines=[]
            for line in t.content.rstrip().splitlines():
                lines.extend(textwrap.wrap(line,width=93,replace_whitespace=False,drop_whitespace=False) or [''])
            story.append(Preformatted('\n'.join(lines),styles['CodePlan']))
        elif t.type=='table_open':
            end=i+1
            while tokens[end].type!='table_close': end+=1
            story.append(table_block(tokens[i:end+1])); story.append(Spacer(1,10))
            i=end
        i+=1
    return story

text=(ROOT/'master_plan.md').read_text(encoding='utf-8')
check=text.split('## 5. Full-credit deliverables and rubric traceability\n',1)[1].split('\n## 6.',1)[0]
check=check.replace('This section is the authoritative completion checklist for this plan. A standalone copy is provided in `deliverables_checklist.md`.','Check off each item only after reviewing the actual evidence. See `master_plan.md` for the complete phased implementation guide.')
(ROOT/'deliverables_checklist.md').write_text('# Full-credit deliverables checklist\n\n'+check,encoding='utf-8')

cover=[]
cover.append(Spacer(1,58))
cover.append(Paragraph('GENERATIVE AI / ASSIGNMENT 2',ParagraphStyle('Kicker',fontName='PlanBold',fontSize=11,textColor=TEAL,spaceAfter=20)))
cover.append(Paragraph('Complete<br/>Master Plan',ParagraphStyle('CoverTitle',fontName='PlanBold',fontSize=38,leading=43,textColor=NAVY,spaceAfter=24)))
cover.append(Paragraph('Clinical criterion matching with a fine-tuned Transformer',ParagraphStyle('Subtitle',fontName='Plan',fontSize=17,leading=23,textColor=INK,spaceAfter=27)))
cover.append(Paragraph('12 sequential phases. Colab-first research compute. Independent local application. Complete rubric traceability.',styles['BodyPlan']))
cover.append(Spacer(1,20))
cover.append(Paragraph('<b>Revised:</b> 9 November 2026, Asia/Karachi<br/><b>Assignment deadline:</b> 10 November 2026; verify Classroom clock time<br/><b>Local device:</b> RTX 3050 Laptop 6 GB / 16 GB RAM / Windows 11<br/><b>Research compute:</b> free Colab, subject to verified allocation',styles['BodyPlan']))
cover.append(Spacer(1,24))
cover.append(Paragraph('<b>Planning status</b><br/>Assignment, local device and Colab service guidance inspected. Dataset structure/split counts verified. No Colab allocation, training run or performance result is claimed. The detailed assignment rubric totals 135 marks despite its 100-mark heading.',styles['BodyPlan']))
cover.append(PageBreak())
cover.append(Paragraph('Contents',styles['H2Plan']))
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOCPlan',fontName='Plan',fontSize=10,leading=14.5,textColor=INK,spaceBefore=5,leftIndent=0,firstLineIndent=0,rightIndent=22)]
toc.dotsMinLevel=0
cover.append(toc)
cover.append(PageBreak())

# Skip title/introduction only; preserve every substantive section from Markdown.
body=text[text.index('## 1. Read this first'):]
dest=PDF_DIR/'GenAI_Assignment_2_Master_Plan.pdf'
PlanDoc(dest).multiBuild(cover+body_story(body))

# Render every final page for layout QA, plus source pages containing the rubric.
doc=fitz.open(dest)
for idx,page in enumerate(doc):
    page.get_pixmap(matrix=fitz.Matrix(1.25,1.25),alpha=False).save(QA_DIR/f'plan-{idx+1:02}.png')
source=fitz.open(ROOT.parent/'GenAI_Assignment#2.pdf')
for idx in [3,4,5,6]:
    source[idx].get_pixmap(matrix=fitz.Matrix(1.2,1.2),alpha=False).save(QA_DIR/f'source-{idx+1}.png')

from PIL import Image, ImageOps, ImageDraw
for start in range(0,len(doc),9):
    sheet=Image.new('RGB',(1200,1710),'#ccd4dc')
    draw=ImageDraw.Draw(sheet)
    for offset in range(min(9,len(doc)-start)):
        im=Image.open(QA_DIR/f'plan-{start+offset+1:02}.png')
        im.thumbnail((385,540))
        x=(offset%3)*400+(400-im.width)//2
        y=(offset//3)*570+22
        sheet.paste(im,(x,y)); draw.text((x,y-17),f'Page {start+offset+1}',fill='black')
    sheet.save(QA_DIR/f'contact-{start//9+1}.png')
print(f'Built {dest}; {len(doc)} pages; {len(text.split())} Markdown words')
print('Page text coverage:',[(i+1,len(p.get_text())) for i,p in enumerate(doc)])
