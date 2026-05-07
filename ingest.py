# pipeline/ingest.py
from pathlib import Path
from dataclasses import asdict, dataclass
print("Importing ingest.py...")
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
import json

@dataclass
class ParsedSection:
    heading: str
    text: str
    page_start: int = 1          # ← NEW: first page this section appears on

@dataclass  
class ParsedTable:
    index: int
    preceding_heading: str
    markdown: str
    page_start: int = 1          # ← NEW: page the table appears on

@dataclass
class ParsedPaper:
    paper_id: str
    sections: list[ParsedSection]
    tables: list[ParsedTable]
    full_markdown: str

def build_converter() -> DocumentConverter:
    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_ocr = False
    pipeline_options.do_table_structure = True
    pipeline_options.table_structure_options.do_cell_matching = True
    pipeline_options.accelerator_options = AcceleratorOptions(
        num_threads=4, device=AcceleratorDevice.CPU
    )
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )

_converter = build_converter()

PARSED_CACHE_DIR = Path("data/parsed_papers")
PARSED_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _page_of(item) -> int:
    """Extract 1-based page number from a Docling item's provenance."""
    try:
        return item.prov[0].page_no
    except (AttributeError, IndexError):
        return 1


def parse_pdf(pdf_path: str | Path) -> ParsedPaper:
    pdf_path = Path(pdf_path)
    cache_path = PARSED_CACHE_DIR / f"{pdf_path.stem}.json"

    if cache_path.exists():
        print(f"  ⚡ Using cached parse for {pdf_path.name}")
        with open(cache_path, "r") as f:
            data = json.load(f)
            return ParsedPaper(
                paper_id=data['paper_id'],
                sections=[ParsedSection(**s) for s in data['sections']],
                tables=[ParsedTable(**t) for t in data['tables']],
                full_markdown=data['full_markdown']
            )

    print(f"  ⚙️  First-time parse for {pdf_path.name} (Loading weights...)")
    result = _converter.convert(str(pdf_path))
    doc = result.document
    print(f"  📄 Parsed {pdf_path.name}")

    # --- collect sections WITH page numbers ---
    sections = []
    current_heading = "preamble"
    current_text = []
    current_page = 1             # page of first item under this heading

    for item, _ in doc.iterate_items():
        t = type(item).__name__
        if t == "SectionHeaderItem":
            if current_text:
                sections.append(ParsedSection(
                    heading=current_heading,
                    text=" ".join(current_text),
                    page_start=current_page,
                ))
            current_heading = item.text
            current_text = []
            current_page = _page_of(item)   # heading's own page
        elif t in ("TextItem", "ListItem"):
            if not current_text:
                current_page = _page_of(item)  # first text item sets the page
            current_text.append(item.text)

    if current_text:
        sections.append(ParsedSection(
            heading=current_heading,
            text=" ".join(current_text),
            page_start=current_page,
        ))

    # --- collect tables WITH page numbers ---
    tables = []
    section_at_table = {}
    heading_cursor = "preamble"
    table_idx = 0

    for item, _ in doc.iterate_items():
        t = type(item).__name__
        if t == "SectionHeaderItem":
            heading_cursor = item.text
        elif t == "TableItem":
            section_at_table[table_idx] = (heading_cursor, _page_of(item))
            table_idx += 1

    for i, table in enumerate(doc.tables):
        heading, page = section_at_table.get(i, ("unknown", 1))
        tables.append(ParsedTable(
            index=i,
            preceding_heading=heading,
            markdown=table.export_to_markdown(doc),
            page_start=page,
        ))

    parsed_paper = ParsedPaper(
        paper_id=pdf_path.stem,
        sections=sections,
        tables=tables,
        full_markdown=doc.export_to_markdown()
    )
    with open(cache_path, "w") as f:
        json.dump(asdict(parsed_paper), f)

    return parsed_paper


def parse_all(papers_dir: str | Path) -> list[ParsedPaper]:
    papers_dir = Path(papers_dir)
    pdfs = list(papers_dir.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs")
    results = []
    for pdf in pdfs:
        print(f"  Parsing {pdf.name}...")
        results.append(parse_pdf(pdf))
    return results