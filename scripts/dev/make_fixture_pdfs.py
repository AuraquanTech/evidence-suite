# scripts/dev/make_fixture_pdfs.py
"""Generate fixture PDFs for testing the pipeline."""

from pathlib import Path


try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
except ImportError:
    print("reportlab not installed. Run: pip install reportlab")
    raise SystemExit(1) from None

OUT = Path("data/fixtures")
OUT.mkdir(parents=True, exist_ok=True)


def write_pdf(path: Path, lines: list[str]) -> None:
    """Write a simple PDF with the given lines."""
    c = canvas.Canvas(str(path), pagesize=letter)
    y = 750
    for line in lines:
        c.drawString(50, y, line)
        y -= 14
        if y < 50:
            c.showPage()
            y = 750
    c.save()


# Good case - clean medical records
good_case = [
    "MRI REPORT",
    "",
    "Patient: Jane Doe",
    "DOB: 1985-03-15",
    "Date of Study: 2025-01-12",
    "",
    "Date of Loss: 2025-01-10",
    "Provider: Dr. Robert Smith, MD",
    "Facility: City Medical Center",
    "",
    "CLINICAL HISTORY:",
    "Motor vehicle accident on 2025-01-10. Patient presents with",
    "neck pain and limited range of motion.",
    "",
    "DIAGNOSIS (ICD-10):",
    "S13.4XXA - Sprain of ligaments of cervical spine, initial encounter",
    "M54.2 - Cervicalgia",
    "",
    "FINDINGS:",
    "1. Disc bulge at C5-C6 with mild foraminal narrowing",
    "2. No evidence of cord compression",
    "3. Mild degenerative changes at C4-C5",
    "",
    "TREATMENT PLAN:",
    "Treatment Start: 2025-01-12",
    "Treatment End: 2025-02-28 (estimated)",
    "",
    "CPT Codes:",
    "97110 - Therapeutic exercises (3x/week x 6 weeks)",
    "97140 - Manual therapy techniques",
    "97012 - Mechanical traction",
    "",
    "IMPRESSION:",
    "Acute cervical strain with disc bulge, likely related to",
    "reported motor vehicle accident. Recommend conservative",
    "treatment with physical therapy.",
    "",
    "___________________________",
    "Dr. Robert Smith, MD",
    "Board Certified Orthopedic Surgeon",
    "License #: MD12345",
]

# Bad case - suspicious/inconsistent records
bad_case = [
    "PATIENT DIARY",
    "",
    "Patient: John Doe",
    "",
    "Date of Loss: 2025-03-20",
    "",
    "Treatment Start: 2025-03-01  (NOTE: This is BEFORE the accident!)",
    "",
    "Diagnosis: 'broken neck' (no ICD-10 code provided)",
    "",
    "No treating provider listed",
    "No medical facility identified",
    "",
    "Symptoms:",
    "- Really bad pain",
    "- Can't do anything",
    "- Worst injury ever",
    "",
    "Treatment received:",
    "- Lots of therapy",
    "- Many appointments",
    "",
    "No imaging studies performed",
    "No objective findings documented",
    "",
    "Billing:",
    "CPT: ???",
    "Charges: $50,000 (no itemization)",
    "",
    "This document was not signed by any medical professional.",
]

# Mixed case - some red flags
mixed_case = [
    "MEDICAL RECORD SUMMARY",
    "",
    "Patient: Bob Johnson",
    "Date of Loss: 2025-02-15",
    "",
    "Provider: Dr. Jane Williams, DC",
    "Facility: Wellness Chiropractic",
    "",
    "DIAGNOSIS:",
    "Cervical strain (no ICD-10 code)",
    "",
    "TREATMENT:",
    "Start: 2025-02-16",
    "Sessions: 45 (daily for 9 weeks)",
    "",
    "CPT: 98941 - Chiropractic manipulative treatment",
    "",
    "NOTE: Treatment frequency exceeds typical guidelines.",
    "No progress notes between sessions.",
    "Same treatment plan for entire duration.",
    "",
    "Imaging: X-ray performed 2025-02-16",
    "Findings: Normal cervical alignment",
    "",
    "Total charges: $22,500",
    "",
    "___________________________",
    "Dr. Jane Williams, DC",
]

write_pdf(OUT / "good_case.pdf", good_case)
write_pdf(OUT / "bad_case.pdf", bad_case)
write_pdf(OUT / "mixed_case.pdf", mixed_case)

print(f"Created fixtures in {OUT.resolve()}:")
print("  - good_case.pdf")
print("  - bad_case.pdf")
print("  - mixed_case.pdf")
