"""Render the plain-language guide without creating temporary preview files."""
from html import escape
from pathlib import Path
import re

from markdown_it import MarkdownIt
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import BaseDocTemplate, Frame, LongTable, PageBreak, PageTemplate, Paragraph, Spacer, TableStyle
from reportlab.platypus.tableofcontents import TableOfContents

ROOT = Path(__file__).resolve().parent
DEST = ROOT / 'pdf' / 'Assignment_Explained_Step_by_Step.pdf'
DEST.parent.mkdir(parents=True, exist_ok=True)
for name, filename in [('Guide','calibri.ttf'), ('GuideBold','calibrib.ttf'), ('GuideItalic','calibrii.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / filename)))
pdfmetrics.registerFontFamily('Guide', normal='Guide', bold='GuideBold', italic='GuideItalic', boldItalic='GuideBold')

NAVY = colors.HexColor('#17324d')
TEAL = colors.HexColor('#087f8c')
INK = colors.HexColor('#263344')
MUTED = colors.HexColor('#607080')
PALE = colors.HexColor('#f0f5f8')
W, H, MARGIN, TOP, BOTTOM = 595.276, 841.89, 52, 56, 48
WIDTH = W - 2 * MARGIN
body = ParagraphStyle('Body', fontName='Guide', fontSize=11, leading=15.2, textColor=INK, spaceAfter=10, allowOrphans=0, allowWidows=0)
bullet = ParagraphStyle('Bullet', parent=body, leftIndent=15, firstLineIndent=-11, spaceAfter=7)
h2 = ParagraphStyle('Section', fontName='GuideBold', fontSize=19, leading=23, textColor=NAVY, spaceAfter=16, keepWithNext=True)
h3 = ParagraphStyle('Subsection', fontName='GuideBold', fontSize=13, leading=17, textColor=TEAL, spaceBefore=8, spaceAfter=8, keepWithNext=True)
cell = ParagraphStyle('Cell', fontName='Guide', fontSize=9.4, leading=12.3, textColor=INK)
head = ParagraphStyle('TableHead', parent=cell, fontName='GuideBold', textColor=colors.white)

def inline(tokens):
    result=[]
    for token in tokens or []:
        if token.type in ('text','code_inline'):
            result.append(escape(token.content))
        elif token.type == 'strong_open': result.append('<b>')
        elif token.type == 'strong_close': result.append('</b>')
        elif token.type == 'em_open': result.append('<i>')
        elif token.type == 'em_close': result.append('</i>')
        elif token.type == 'link_open':
            href=token.attrGet('href') or ''
            result.append('<a href="'+escape(href,quote=True)+'" color="#087f8c">' if href.startswith('https://') else '<font color="#087f8c">')
        elif token.type == 'link_close':
            # Links within this guide point to separate local companion files.
            # ReportLab has no portable relative-file link here.
            result.append('</a>' if result and any(x.startswith('<a href=') for x in result) and not any(x=='</a>' for x in result) else '</font>')
        elif token.type in ('softbreak','hardbreak'): result.append(' ')
    return ''.join(result)

class GuideDoc(BaseDocTemplate):
    def __init__(self, filename):
        super().__init__(str(filename), pagesize=(W,H), title='Your assignment, explained step by step', author='Assignment planning guide')
        frame=Frame(MARGIN,BOTTOM,WIDTH,H-TOP-BOTTOM,id='main',leftPadding=0,rightPadding=0,topPadding=0,bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='pages', frames=frame, onPage=self.chrome))
    def chrome(self, canvas, doc):
        if doc.page==1: return
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor('#d9e3e9'))
        canvas.line(MARGIN,H-35,W-MARGIN,H-35)
        canvas.setFont('Guide',8.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN,H-26,'GEN AI ASSIGNMENT 2  /  PLAIN-LANGUAGE COMPANION')
        canvas.drawString(MARGIN,26,'Explanatory guide  |  Examples are illustrative')
        canvas.drawRightString(W-MARGIN,26,str(doc.page))
        canvas.restoreState()
    def afterFlowable(self, flowable):
        if isinstance(flowable,Paragraph) and flowable.style.name=='Section' and flowable.getPlainText()!='Contents':
            title=flowable.getPlainText()
            key='section_'+re.sub(r'[^a-zA-Z0-9]+','_',title)
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(title,key,0,False)
            self.notify('TOCEntry',(0,title,self.page,key))

def make_table(tokens):
    rows=[]; row=[]; in_header=False
    for token in tokens:
        if token.type=='thead_open': in_header=True
        elif token.type=='thead_close': in_header=False
        elif token.type=='tr_open': row=[]
        elif token.type=='inline': row.append(Paragraph(inline(token.children),head if in_header else cell))
        elif token.type=='tr_close': rows.append(row)
    widths=[WIDTH*0.31, WIDTH*0.23, WIDTH*0.46]
    table=LongTable(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([
        ('BACKGROUND',(0,0),(-1,0),NAVY),('VALIGN',(0,0),(-1,-1),'TOP'),
        ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,PALE]),
        ('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),
        ('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
    ]))
    return table

def make_body(text):
    tokens=MarkdownIt('commonmark').enable('table').parse(text)
    story=[]; lists=[]; i=0
    while i<len(tokens):
        token=tokens[i]
        if token.type=='heading_open':
            level=int(token.tag[1:]); title=tokens[i+1]
            if level==1: i+=3; continue
            if level==2:
                if story: story.append(PageBreak())
                story.append(Paragraph(inline(title.children),h2))
            else: story.append(Paragraph(inline(title.children),h3))
            i+=3; continue
        if token.type in ('bullet_list_open','ordered_list_open'):
            lists.append({'ordered':token.type=='ordered_list_open','n':int(token.attrGet('start') or 1)})
        elif token.type in ('bullet_list_close','ordered_list_close'): lists.pop()
        elif token.type=='paragraph_open':
            content=inline(tokens[i+1].children)
            if lists:
                context=lists[-1]
                prefix=str(context['n'])+'. ' if context['ordered'] else '- '
                context['n']+=1
                story.append(Paragraph(prefix+content,bullet))
            else: story.append(Paragraph(content,body))
            i+=3; continue
        elif token.type=='table_open':
            end=i+1
            while tokens[end].type!='table_close': end+=1
            story.append(make_table(tokens[i:end+1])); story.append(Spacer(1,10)); i=end
        i+=1
    return story

cover=[Spacer(1,75),
    Paragraph('GENERATIVE AI / ASSIGNMENT 2',ParagraphStyle('Kicker',fontName='GuideBold',fontSize=11,textColor=TEAL,spaceAfter=23)),
    Paragraph('Your assignment,<br/>explained step by step',ParagraphStyle('Title',fontName='GuideBold',fontSize=32,leading=38,textColor=NAVY,spaceAfter=25)),
    Paragraph('A plain-language companion to the master plan',ParagraphStyle('Subtitle',fontName='Guide',fontSize=17,leading=23,textColor=INK,spaceAfter=25)),
    Paragraph('Understand what we are building, why each phase matters, and what you should see when it is complete.',body),
    Spacer(1,18),
    Paragraph('<b>Our working approach</b><br/>Google Colab for training and experiments, subject to GPU/session availability; your laptop for the final browser application.',body),
    Spacer(1,18),
    Paragraph('<b>How to use this guide</b><br/>Read for understanding, then use the matching phase in the technical master plan for implementation and checks.',body),
    PageBreak(), Paragraph('Contents',h2)]
toc=TableOfContents()
toc.levelStyles=[ParagraphStyle('TOC',fontName='Guide',fontSize=10.4,leading=14,textColor=INK,spaceBefore=4,rightIndent=24)]
toc.dotsMinLevel=0
cover.extend([toc,PageBreak()])
markdown=(ROOT/'plain_language_guide.md').read_text(encoding='utf-8')
GuideDoc(DEST).multiBuild(cover+make_body(markdown))
print('Created:',DEST)
