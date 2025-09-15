from io import BytesIO
from reportlab.lib.pagesizes import LETTER
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib import colors


def build_feedback_pdf(summary_text: str) -> bytes:
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=LETTER, leftMargin=0.8*inch, rightMargin=0.8*inch, topMargin=0.8*inch, bottomMargin=0.8*inch)
    styles = getSampleStyleSheet()
    story = []

    title_style = styles['Title']
    title_style.textColor = colors.HexColor('#111827')
    story.append(Paragraph('Interview Feedback Summary', title_style))
    story.append(Spacer(1, 0.3*inch))

    body_style = styles['BodyText']
    for line in (summary_text or '').split('\n'):
        story.append(Paragraph(line.replace('  ', '&nbsp;&nbsp;'), body_style))
        story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    pdf_bytes = buf.getvalue()
    buf.close()
    return pdf_bytes
