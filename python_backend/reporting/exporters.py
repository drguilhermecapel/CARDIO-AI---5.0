import uuid
from datetime import datetime
from typing import Dict, Any, List
import xml.etree.ElementTree as ET
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from jinja2 import Template
import io
import json

class HL7CDAExporter:
    """
    Generates HL7 CDA R2 XML documents for ECG reports.
    """
    def generate(self, data: Dict[str, Any]) -> str:
        patient = data.get('patient', {})
        findings = data.get('findings', {})
        meta = data.get('metadata', {})
        
        # Root OID for the organization (Mock)
        org_oid = "2.16.840.1.113883.3.12345"
        
        root = ET.Element("ClinicalDocument", xmlns="urn:hl7-org:v3")
        
        # Header
        ET.SubElement(root, "realmCode", code="US")
        ET.SubElement(root, "typeId", root="2.16.840.1.113883.1.3", extension="POCD_HD000040")
        ET.SubElement(root, "id", root=str(uuid.uuid4()))
        ET.SubElement(root, "code", code="11524-6", codeSystem="2.16.840.1.113883.6.1", displayName="EKG Study")
        ET.SubElement(root, "title").text = "CardioAI Nexus ECG Report"
        ET.SubElement(root, "effectiveTime", value=datetime.now().strftime("%Y%m%d%H%M%S"))
        ET.SubElement(root, "confidentialityCode", code="N", codeSystem="2.16.840.1.113883.5.25")
        
        # Patient
        record_target = ET.SubElement(root, "recordTarget")
        patient_role = ET.SubElement(record_target, "patientRole")
        ET.SubElement(patient_role, "id", extension=patient.get('id', 'UNKNOWN'), root=org_oid)
        pat_node = ET.SubElement(patient_role, "patient")
        name_node = ET.SubElement(pat_node, "name")
        ET.SubElement(name_node, "given").text = patient.get('name', 'Unknown')
        ET.SubElement(pat_node, "administrativeGenderCode", code=patient.get('sex', 'U')[0], codeSystem="2.16.840.1.113883.5.1")
        ET.SubElement(pat_node, "birthTime", value=patient.get('dob', ''))
        
        # Author (Software)
        author = ET.SubElement(root, "author")
        ET.SubElement(author, "time", value=datetime.now().strftime("%Y%m%d%H%M%S"))
        assigned_author = ET.SubElement(author, "assignedAuthor")
        ET.SubElement(assigned_author, "id", root=org_oid, extension="CardioAI_v5.0")
        author_dev = ET.SubElement(assigned_author, "assignedAuthoringDevice")
        ET.SubElement(author_dev, "manufacturerModelName").text = f"CardioAI Nexus v{meta.get('sw_version', '5.0')}"
        ET.SubElement(author_dev, "softwareName").text = "ECG Analysis Engine"
        
        # Body
        component = ET.SubElement(root, "component")
        structured_body = ET.SubElement(component, "structuredBody")
        
        # Section: Findings
        comp_findings = ET.SubElement(structured_body, "component")
        sect_findings = ET.SubElement(comp_findings, "section")
        ET.SubElement(sect_findings, "code", code="59776-5", codeSystem="2.16.840.1.113883.6.1", displayName="Findings")
        ET.SubElement(sect_findings, "title").text = "Automated Findings"
        text_findings = ET.SubElement(sect_findings, "text")
        
        # Diagnosis List
        list_node = ET.SubElement(text_findings, "list")
        ET.SubElement(list_node, "item").text = f"Diagnosis: {findings.get('diagnosis', 'Normal')}"
        ET.SubElement(list_node, "item").text = f"Confidence: {findings.get('confidence', 0.0):.2f}"
        
        # Metrics
        metrics = findings.get('metrics', {})
        ET.SubElement(list_node, "item").text = f"HR: {metrics.get('hr', 'N/A')} bpm"
        ET.SubElement(list_node, "item").text = f"QTc: {metrics.get('qtc', 'N/A')} ms"
        
        return ET.tostring(root, encoding='unicode')

class PDFExporter:
    """
    Generates a professional PDF report.
    """
    def generate(self, data: Dict[str, Any]) -> bytes:
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        styles = getSampleStyleSheet()
        story = []
        
        # Title
        story.append(Paragraph("CardioAI Nexus - ECG Report", styles['Title']))
        story.append(Spacer(1, 12))
        
        # Patient Info
        patient = data.get('patient', {})
        meta = data.get('metadata', {})
        
        p_info = [
            [f"Patient ID: {patient.get('id')}", f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}"],
            [f"Name: {patient.get('name')}", f"DOB: {patient.get('dob')}"],
            [f"Sex: {patient.get('sex')}", f"Device: {meta.get('device_id')}"]
        ]
        
        t = Table(p_info, colWidths=[300, 200])
        t.setStyle(TableStyle([
            ('ALIGN', (0,0), (-1,-1), 'LEFT'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('FONTSIZE', (0,0), (-1,-1), 10),
            ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 20))
        
        # Findings
        findings = data.get('findings', {})
        story.append(Paragraph("Automated Interpretation", styles['Heading2']))
        
        diagnosis_style = ParagraphStyle('Diagnosis', parent=styles['Normal'], fontSize=14, textColor=colors.darkblue)
        story.append(Paragraph(f"Diagnosis: {findings.get('diagnosis', 'Unconfirmed')}", diagnosis_style))
        story.append(Paragraph(f"Confidence: {findings.get('confidence', 0.0):.1%}", styles['Normal']))
        story.append(Spacer(1, 10))
        
        # Metrics Table
        metrics = findings.get('metrics', {})
        m_data = [
            ['Metric', 'Value', 'Reference'],
            ['Heart Rate', f"{metrics.get('hr', '--')} bpm", "60-100"],
            ['PR Interval', f"{metrics.get('pr', '--')} ms", "120-200"],
            ['QRS Duration', f"{metrics.get('qrs', '--')} ms", "80-120"],
            ['QT / QTc', f"{metrics.get('qt', '--')} / {metrics.get('qtc', '--')} ms", "< 450 (M) / 460 (F)"],
            ['Axis (P/QRS/T)', f"{metrics.get('axis', '--')}", "-30 to +90"]
        ]
        
        mt = Table(m_data, colWidths=[150, 150, 150])
        mt.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
            ('TEXTCOLOR', (0,0), (-1,0), colors.black),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
            ('GRID', (0,0), (-1,-1), 1, colors.black),
        ]))
        story.append(mt)
        story.append(Spacer(1, 20))
        
        # Signature
        story.append(Spacer(1, 30))
        story.append(Paragraph(f"Digitally Signed by: CardioAI Engine v{meta.get('sw_version')}", styles['Italic']))
        story.append(Paragraph(f"Signature Hash: {data.get('signature', 'PENDING')[:32]}...", styles['Italic']))
        
        doc.build(story)
        return buffer.getvalue()

class HTMLReporter:
    """
    Generates HTML report for web viewing.
    """
    TEMPLATE = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>ECG Report - {{ patient.id }}</title>
        <style>
            body { font-family: sans-serif; margin: 40px; }
            .header { border-bottom: 2px solid #333; padding-bottom: 10px; margin-bottom: 20px; }
            .section { margin-bottom: 20px; }
            .metric-box { display: inline-block; width: 150px; padding: 10px; background: #f0f0f0; margin-right: 10px; border-radius: 5px; }
            .diagnosis { font-size: 1.5em; color: #0056b3; font-weight: bold; }
            .alert { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>CardioAI Nexus Report</h1>
            <p>Patient: {{ patient.name }} ({{ patient.sex }}, {{ patient.age }}y) | ID: {{ patient.id }}</p>
            <p>Date: {{ metadata.timestamp }} | Device: {{ metadata.device_id }}</p>
        </div>
        
        <div class="section">
            <h2>Interpretation</h2>
            <p class="diagnosis">{{ findings.diagnosis }}</p>
            <p>Confidence: {{ "%.1f"|format(findings.confidence * 100) }}%</p>
            {% if findings.is_abnormal %}
            <p class="alert">⚠️ Abnormal Findings Detected</p>
            {% endif %}
        </div>
        
        <div class="section">
            <h2>Key Metrics</h2>
            <div class="metric-box">
                <strong>HR</strong><br>{{ findings.metrics.hr }} bpm
            </div>
            <div class="metric-box">
                <strong>QRS</strong><br>{{ findings.metrics.qrs }} ms
            </div>
            <div class="metric-box">
                <strong>QTc</strong><br>{{ findings.metrics.qtc }} ms
            </div>
            <div class="metric-box">
                <strong>PR</strong><br>{{ findings.metrics.pr }} ms
            </div>
        </div>
        
        <div class="section">
            <small>Software Version: {{ metadata.sw_version }} | Signature: {{ signature[:16] }}...</small>
        </div>
    </body>
    </html>
    """
    
    def generate(self, data: Dict[str, Any]) -> str:
        t = Template(self.TEMPLATE)
        return t.render(**data)
