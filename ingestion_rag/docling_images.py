import logging
import time
from pathlib import Path
from docling_core.types.doc import PictureItem, TableItem
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption

logging.basicConfig(level=logging.INFO)
input_pdf = Path("test.pdf")  # Replace with your PDF path
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

pipeline_options = PdfPipelineOptions()
pipeline_options.images_scale = 2.0
pipeline_options.generate_page_images = True
pipeline_options.generate_picture_images = True

converter = DocumentConverter(
    format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
)
conv_res = converter.convert(input_pdf)
doc_filename = conv_res.input.file.stem

# Save page images
for page_no, page in conv_res.document.pages.items():
    page_image_path = output_dir / f"{doc_filename}-page-{page_no}.png"
    page.image.pil_image.save(page_image_path, format="PNG")

# Save picture and table images
picture_counter = table_counter = 0
for element, _ in conv_res.document.iterate_items():
    if isinstance(element, PictureItem):
        picture_counter += 1
        img_path = output_dir / f"{doc_filename}-picture-{picture_counter}.png"
        element.get_image(conv_res.document).save(img_path, "PNG")
    elif isinstance(element, TableItem):
        table_counter += 1
        img_path = output_dir / f"{doc_filename}-table-{table_counter}.png"
        element.get_image(conv_res.document).save(img_path, "PNG")

logging.info(f"Images saved to {output_dir}")