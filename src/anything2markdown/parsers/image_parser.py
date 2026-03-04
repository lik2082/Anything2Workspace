"""Parser for image files using Vision-Language Models (VLM)."""

import base64
from datetime import datetime
from pathlib import Path

import httpx
import structlog
from openai import OpenAI

from ..config import settings
from ..schemas.result import ParseResult
from ..utils.file_utils import flatten_path
from .base import BaseParser

logger = structlog.get_logger(__name__)

# --- Specialized Prompts ---

PROMPT_AUTO = """
Analyze this image and generate a structured markdown description.
Based on the content, identify its type and extract relevant information:

1. **Decorative/Artistic**: If the image is purely decorative or artistic, provide a brief description of the style, mood, and content.
2. **Informative (Text/Process)**: If the image contains text, diagrams, or flowcharts, transcribe the text faithfully and describe the process or information in detail.
3. **Pattern/Entity Recognition**: If the image contains specific entities (people, landmarks, objects), identify them and classify them (e.g., "Portrait of Person X", "Landmark Y").

Output ONLY the markdown description. Do not include conversational filler.
"""

PROMPT_DECORATIVE = """
This is a decorative or artistic image.
Focus on describing the visual style, color palette, mood, composition, and artistic elements.
Do not transcribe text unless it is a central part of the artwork.
Provide a rich, evocative description suitable for understanding the image's aesthetic role.
Output ONLY the markdown description.
"""

PROMPT_INFORMATIVE = """
This is an informative image (document, diagram, screenshot, or chart).
Your primary task is to EXTRACT INFORMATION faithfully.
1. Transcribe all visible text (OCR).
2. Describe any diagrams, flowcharts, or relationships between elements.
3. If it is a data chart, summarize the data points and trends.
Ignore artistic style or mood. Focus on content.
Output ONLY the markdown description.
"""

PROMPT_ENTITY = """
This is an image for entity or pattern recognition.
Identify specific people, landmarks, biological species, products, or objects.
1. Classify the main subject (e.g., "Eiffel Tower", "Golden Retriever", "iPhone 15").
2. Provide key details about the identified entity.
3. If multiple entities are present, list them.
Output ONLY the markdown description.
"""

# Mapping directory names to prompts
DIR_PROMPT_MAP = {
    "art": PROMPT_DECORATIVE,
    "decorative": PROMPT_DECORATIVE,
    "style": PROMPT_DECORATIVE,
    
    "info": PROMPT_INFORMATIVE,
    "doc": PROMPT_INFORMATIVE,
    "ocr": PROMPT_INFORMATIVE,
    "text": PROMPT_INFORMATIVE,
    
    "entity": PROMPT_ENTITY,
    "pattern": PROMPT_ENTITY,
    "id": PROMPT_ENTITY,
}

class ImageParser(BaseParser):
    """
    Parser for image files (.jpg, .png, etc.) using a Vision-Language Model.
    Supports:
    - Decorative images (description)
    - Informative images (OCR + layout)
    - Entity images (recognition)
    """

    supported_extensions = [
        ".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff"
    ]
    parser_name = "image_vlm"

    def __init__(self):
        """Initialize the OpenAI client for VLM API."""
        self.client = None
        if settings.siliconflow_api_key:
            self.client = OpenAI(
                api_key=settings.siliconflow_api_key,
                base_url=settings.siliconflow_base_url,
                timeout=60.0,
                max_retries=2,
            )

    def can_handle(self, file_path: Path) -> bool:
        """Check if file extension is supported."""
        return file_path.suffix.lower() in self.supported_extensions

    def parse(self, file_path: Path, output_dir: Path) -> ParseResult:
        """
        Parse an image file using VLM.

        Args:
            file_path: Path to the input image
            output_dir: Directory to save output

        Returns:
            ParseResult with conversion details
        """
        started_at = datetime.now()

        if not self.client:
            completed_at = datetime.now()
            logger.error("ImageParser: no API key configured")
            return ParseResult(
                source_path=file_path,
                output_path=Path(""),
                source_type="file",
                parser_used=self.parser_name,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                output_format="markdown",
                error_message="No API key configured (set SILICONFLOW_API_KEY)",
            )

        # Determine prompt based on directory name
        # Check if parent directory name (case-insensitive) matches any keyword
        parent_dir = file_path.parent.name.lower()
        
        # Default to auto-detect prompt
        prompt = PROMPT_AUTO
        prompt_type = "auto"

        # Check for specific intent folders
        for key, specific_prompt in DIR_PROMPT_MAP.items():
            if key in parent_dir:
                prompt = specific_prompt
                prompt_type = key
                break

        logger.info(
            "ImageParser parsing", 
            file=file_path.name, 
            model=settings.vision_model,
            intent=prompt_type
        )

        try:
            # Read and encode image
            with open(file_path, "rb") as image_file:
                base64_image = base64.b64encode(image_file.read()).decode("utf-8")

            # Determine mime type
            mime_type = "image/jpeg"
            if file_path.suffix.lower() == ".png":
                mime_type = "image/png"
            elif file_path.suffix.lower() == ".webp":
                mime_type = "image/webp"
            elif file_path.suffix.lower() == ".gif":
                mime_type = "image/gif"

            # Call VLM
            response = self.client.chat.completions.create(
                model=settings.vision_model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}",
                                },
                            },
                        ],
                    }
                ],
                max_tokens=4000,
                temperature=0.2,
            )
            
            content = response.choices[0].message.content
            
            if not content:
                raise ValueError("Empty response from VLM")

            # Generate flattened output filename
            output_name = flatten_path(file_path, settings.input_dir) + ".md"
            output_path = output_dir / output_name
            
            # Ensure parent directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write output
            output_path.write_text(content, encoding="utf-8")

            completed_at = datetime.now()

            logger.info(
                "ImageParser complete",
                file=file_path.name,
                chars=len(content),
                duration=f"{(completed_at - started_at).total_seconds():.1f}s",
            )

            return ParseResult(
                source_path=file_path,
                output_path=output_path,
                source_type="file",
                parser_used=self.parser_name,
                status="success",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                output_format="markdown",
                character_count=len(content),
                metadata={
                    "model": settings.vision_model,
                },
            )

        except Exception as e:
            completed_at = datetime.now()
            logger.error("ImageParser failed", file=file_path.name, error=str(e))

            return ParseResult(
                source_path=file_path,
                output_path=Path(""),
                source_type="file",
                parser_used=self.parser_name,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                duration_seconds=(completed_at - started_at).total_seconds(),
                output_format="markdown",
                error_message=str(e),
            )
