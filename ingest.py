# pipeline/ingest.py
from pathlib import Path
from dataclasses import asdict, dataclass
print("Importing ingest.py...")
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
import json
import base64
import fitz  # PyMuPDF
@dataclass
class ParsedFigure:
    index: int
    caption: str
    page_start: int = 1
    image_path: str = ""   # stored artifact path

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
    figures: list[ParsedFigure]   # NEW
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
FIGURE_DIR = PARSED_CACHE_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def _page_of(item) -> int:
    """Extract 1-based page number from a Docling item's provenance."""
    try:
        return item.prov[0].page_no
    except (AttributeError, IndexError):
        return 1


def parse_pdf(pdf_path: str | Path, converter=None) -> ParsedPaper:
    converter = converter or _converter
    pdf_path = Path(pdf_path)
    cache_path = PARSED_CACHE_DIR / f"{pdf_path.stem}.json"
    pdf_doc = fitz.open(str(pdf_path))
    if cache_path.exists():
        print(f"  ⚡ Using cached parse for {pdf_path.name}")
        with open(cache_path, "r") as f:
            data = json.load(f)
            return ParsedPaper(
                paper_id=data['paper_id'],
                sections=[ParsedSection(**s) for s in data['sections']],
                tables=[ParsedTable(**t) for t in data['tables']],
                full_markdown=data['full_markdown'],
                figures=[ParsedFigure(**f) for f in data.get("figures", [])],
            )

    print(f"  ⚙️  First-time parse for {pdf_path.name} (Loading weights...)")
    result = converter.convert(str(pdf_path))
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

    figures = []
    figure_idx = 0

    for item, _ in doc.iterate_items():
        t = type(item).__name__

        if t in ("PictureItem", "ImageItem"):
            page = _page_of(item) - 1

            caption = ""

            bbox = None
            try:
                bbox = item.prov[0].bbox
            except Exception:
                bbox = None

            image_path = ""

            if bbox:
                page_obj = pdf_doc.load_page(page)

                x0, y0, x1, y1 = bbox.l, bbox.t, bbox.r, bbox.b

                if x1 < x0:
                    x0, x1 = x1, x0
                if y1 < y0:
                    y0, y1 = y1, y0

                if str(getattr(bbox, "coord_origin", "")).endswith("BOTTOMLEFT"):
                    h = page_obj.rect.height
                    y0, y1 = h - bbox.b, h - bbox.t

                rect = fitz.Rect(x0, y0, x1, y1)

                if rect.width <= 1 or rect.height <= 1:
                    rect = page_obj.rect

                pix = page_obj.get_pixmap(clip=rect, dpi=200)

                image_path = str(FIGURE_DIR / f"{pdf_path.stem}_fig{figure_idx}.png")
                pix.save(image_path)

            figures.append(ParsedFigure(
                index=figure_idx,
                caption=caption,
                page_start=page + 1,
                image_path=image_path
            ))

            figure_idx += 1


    parsed_paper = ParsedPaper(
        paper_id=pdf_path.stem,
        sections=sections,
        tables=tables,
        full_markdown=doc.export_to_markdown(),
        figures=figures
    )
    with open(cache_path, "w") as f:
        json.dump(asdict(parsed_paper), f)
    pdf_doc.close()
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