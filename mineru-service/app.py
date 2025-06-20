import os
import asyncio
import tempfile
import shutil
import subprocess
from typing import Optional, Dict, Any, List
from datetime import datetime
import json
import base64
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import aiofiles
from loguru import logger
import redis
from tenacity import retry, stop_after_attempt, wait_exponential

# Check MinerU availability using command line tool (official method)
def check_mineru_availability():
    try:
        result = subprocess.run(['mineru', '--version'], capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            version = result.stdout.strip() if result.stdout else "Unknown"
            logger.info(f"✅ MinerU command available: {version}")
            return True, version
        else:
            logger.warning(f"❌ MinerU command failed: {result.stderr}")
            return False, "Command failed"
    except subprocess.TimeoutExpired:
        logger.warning("❌ MinerU command timeout (might be downloading models)")
        # Return True if command times out - likely downloading models on first run
        logger.info("⚠️ MinerU command timed out but proceeding anyway")
        return True, "Available (timed out - likely downloading models)"
    except FileNotFoundError:
        logger.warning("❌ MinerU command not found")
        return False, "Not found"
    except Exception as e:
        logger.warning(f"❌ MinerU check error: {e}")
        return False, str(e)

MINERU_AVAILABLE, MINERU_VERSION = check_mineru_availability()

# Configuration
class Settings(BaseModel):
    api_key: Optional[str] = os.getenv("MINERU_API_KEY")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")
    max_file_size: int = int(os.getenv("MAX_FILE_SIZE", "50")) * 1024 * 1024  # 50MB default
    cache_ttl: int = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour default
    debug: bool = os.getenv("DEBUG", "false").lower() == "true"
    
    # 🚀 NEW: SGLang client settings
    mineru_device: str = os.getenv("MINERU_DEVICE", "cuda")  # Default to cuda for SGLang
    mineru_backend: str = os.getenv("MINERU_BACKEND", "vlm-sglang-client")  # Default to SGLang client
    sglang_url: str = os.getenv("SGLANG_URL", "http://127.0.0.1:8001")  # SGLang server URL
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "/app/models")

settings = Settings()

# FastAPI app
app = FastAPI(
    title="MinerU PDF Processing Service",
    description="Advanced PDF processing service for research papers using MinerU with SGLang acceleration",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer(auto_error=False)

# Redis client
try:
    redis_client = redis.from_url(settings.redis_url)
    redis_client.ping()
    logger.info("Redis connection established")
except Exception as e:
    logger.warning(f"Redis not available: {e}")
    redis_client = None

# Models
class ProcessingOptions(BaseModel):
    # Core MinerU options (matching CLI parameters from README)
    method: str = Field(default="auto", description="Processing method: auto, txt, ocr")
    backend: str = Field(default=settings.mineru_backend, description="Backend: pipeline, vlm-transformers, vlm-sglang-engine, vlm-sglang-client")  # Updated with sglang-client
    lang: Optional[str] = Field(default=None, description="Document language for OCR accuracy")
    start_page: Optional[int] = Field(default=None, description="Starting page (0-based)")
    end_page: Optional[int] = Field(default=None, description="Ending page (0-based)")
    formula: bool = Field(default=True, description="Enable formula parsing")
    table: bool = Field(default=True, description="Enable table parsing")
    device: str = Field(default=settings.mineru_device, description="Inference device: cpu, cuda, cuda:0, npu, mps")  # Use env default
    vram: Optional[int] = Field(default=None, description="Max GPU VRAM usage per process (MB)")
    model_source: str = Field(default="huggingface", description="Model source: huggingface, modelscope, local")
    
    # 🚀 NEW: SGLang client options
    sglang_url: str = Field(default=settings.sglang_url, description="SGLang server URL")
    
    # Output options
    output_format: str = Field(default="both", description="Output format: json, markdown, or both")
    extract_images: bool = Field(default=True, description="Extract images from PDF")
    
    # Legacy aliases for backwards compatibility
    enableOCR: Optional[bool] = Field(default=None, description="Legacy: use method='ocr' instead")
    layoutAnalysis: Optional[bool] = Field(default=None, description="Legacy: always enabled")
    extractImages: Optional[bool] = Field(default=None, description="Legacy: use extract_images")
    outputFormat: Optional[str] = Field(default=None, description="Legacy: use output_format")
    
    force: bool = Field(default=False, description="Force reprocessing, ignore cache")

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class ElementMetadata(BaseModel):
    fontSize: Optional[float] = None
    fontWeight: Optional[str] = None
    isItalic: Optional[bool] = None
    imageData: Optional[str] = None
    figureCaption: Optional[str] = None
    tableCaption: Optional[str] = None
    equationType: Optional[str] = None

class Hierarchy(BaseModel):
    level: int
    parent: Optional[str] = None
    section: str

class MinerUElement(BaseModel):
    type: str
    content: str
    bbox: BoundingBox
    pageNumber: int
    hierarchy: Optional[Hierarchy] = None
    confidence: float
    metadata: Optional[ElementMetadata] = None

class Section(BaseModel):
    title: str
    level: int
    content: str
    elements: List[MinerUElement]

class StructuredContent(BaseModel):
    title: Optional[str] = None
    abstract: Optional[str] = None
    sections: List[Section]

class ProcessingMetadata(BaseModel):
    totalPages: int
    processingTimeMs: int
    detectedLanguage: str
    documentType: str

class ProcessingResponse(BaseModel):
    success: bool
    elements: List[MinerUElement]
    markdown: str
    structuredContent: StructuredContent
    metadata: ProcessingMetadata
    error: Optional[str] = None

# Authentication
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if settings.api_key and (not credentials or credentials.credentials != settings.api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# 🚀 SGLang server health check
async def check_sglang_server(url: str) -> bool:
    """Check if SGLang server is available"""
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(f"{url}/get_server_info", timeout=5) as response:
                return response.status == 200
    except:
        try:
            # Fallback to curl/requests
            import requests
            response = requests.get(f"{url}/get_server_info", timeout=5)
            return response.status_code == 200
        except:
            return False

# MinerU Processor using official command-line interface
class MinerUProcessor:
    def __init__(self):
        self.temp_dir = Path("/tmp/mineru_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
    def _build_mineru_command(self, input_path: str, output_dir: str, options: ProcessingOptions) -> List[str]:
        """Build MinerU command based on options"""
        cmd = ['mineru', '-p', input_path, '-o', output_dir]
        
        # Add method if specified
        if options.method and options.method != "auto":
            cmd.extend(['-m', options.method])
        
        # 🚀 Add backend (prioritize SGLang client)
        if options.backend and options.backend != "pipeline":
            cmd.extend(['-b', options.backend])
            
            # 🚀 Add SGLang URL for client mode
            if options.backend == "vlm-sglang-client":
                cmd.extend(['-u', options.sglang_url])
                logger.info(f"🚀 Using SGLang client with server: {options.sglang_url}")
            
        # Add language for OCR accuracy
        if options.lang:
            cmd.extend(['-l', options.lang])
            
        # Add page range
        if options.start_page is not None:
            cmd.extend(['-s', str(options.start_page)])
        if options.end_page is not None:
            cmd.extend(['-e', str(options.end_page)])
            
        # Add formula parsing
        if not options.formula:
            cmd.extend(['-f', 'false'])
            
        # Add table parsing  
        if not options.table:
            cmd.extend(['-t', 'false'])
            
        # Add device (only needed for non-client backends)
        if options.device and options.device != "cpu" and options.backend != "vlm-sglang-client":
            cmd.extend(['-d', options.device])
            
        # Add VRAM limit (only for non-client backends)
        if options.vram and options.backend != "vlm-sglang-client":
            cmd.extend(['--vram', str(options.vram)])
            
        # Add model source
        if options.model_source and options.model_source != "huggingface":
            cmd.extend(['--source', options.model_source])
            
        return cmd
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_pdf(self, pdf_path: str, options: ProcessingOptions) -> ProcessingResponse:
        """Process PDF using MinerU command-line tool"""
        start_time = datetime.now()
        
        if not MINERU_AVAILABLE:
            raise HTTPException(status_code=503, detail="MinerU not available")
        
        # 🚀 Check SGLang server if using client mode
        if options.backend == "vlm-sglang-client":
            sglang_available = await check_sglang_server(options.sglang_url)
            if not sglang_available:
                logger.warning(f"SGLang server not available at {options.sglang_url}, falling back to vlm-sglang-engine")
                options.backend = "vlm-sglang-engine"
            else:
                logger.info(f"✅ SGLang server available at {options.sglang_url}")
        
        # Create temporary output directory
        output_dir = self.temp_dir / f"output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Build and execute MinerU command
            cmd = self._build_mineru_command(pdf_path, str(output_dir), options)
            logger.info(f"Executing command: {' '.join(cmd)}")
            
            # Run MinerU command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(output_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)  # 5 minute timeout
            
            # Log stdout and stderr for debugging
            if stdout:
                logger.info(f"MinerU stdout: {stdout.decode()}")
            if stderr:
                logger.warning(f"MinerU stderr: {stderr.decode()}")
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                logger.error(f"MinerU command failed with return code {process.returncode}: {error_msg}")
                raise ValueError(f"MinerU processing failed (code {process.returncode}): {error_msg}")
            
            # Parse MinerU output
            result = await self._parse_mineru_output(output_dir, options)
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            result.metadata.processingTimeMs = int(processing_time)
            
            logger.info(f"🚀 Processing completed in {processing_time:.0f}ms using backend: {options.backend}")
            return result
            
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Processing timeout")
        except Exception as e:
            # Enhanced error logging
            if hasattr(e, 'last_attempt') and hasattr(e.last_attempt, 'exception'):
                underlying_error = e.last_attempt.exception()
                logger.error(f"Processing error (after {e.last_attempt.attempt_number} retries): {underlying_error}")
                if isinstance(underlying_error, HTTPException):
                    raise underlying_error
                else:
                    raise HTTPException(status_code=500, detail=f"Processing failed: {str(underlying_error)}")
            else:
                logger.error(f"Processing error: {e}")
                raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
        finally:
            # Cleanup temporary output directory
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
    
 
    async def _parse_mineru_output(self, output_dir: Path, options: ProcessingOptions) -> ProcessingResponse:
        """Parse MinerU output files into our response format"""
        elements = []
        markdown_content = ""
        sections = []
        extracted_title = None
        extracted_abstract = None
        
        try:
            # Log all files in output directory for debugging
            all_files = list(output_dir.rglob("*"))
            logger.info(f"Files in output directory {output_dir}:")
            for file in all_files:
                logger.info(f"  {file.relative_to(output_dir)} ({'dir' if file.is_dir() else 'file'})")
            
            # Look for MinerU output files (typically JSON and markdown)
            json_files = list(output_dir.rglob("*.json"))  # Use rglob to search subdirectories
            md_files = list(output_dir.rglob("*.md"))
            
            # Prioritize middle.json files which contain the structured data with bounding boxes
            middle_json_files = list(output_dir.rglob("*_middle.json"))
            model_json_files = list(output_dir.rglob("*_model.json"))
            if not model_json_files:
                # Fallback for model output with .txt extension, as seen in some versions
                model_json_files = list(output_dir.rglob("*_model_output.txt"))
            content_list_files = list(output_dir.rglob("*_content_list.json"))
            
            logger.info(f"Found {len(json_files)} JSON files, {len(middle_json_files)} middle files, {len(model_json_files)} model files, {len(content_list_files)} content_list files, and {len(md_files)} MD files")
            
            # Parse middle.json output if available (preferred for structured data)
            if middle_json_files:
                json_file = middle_json_files[0]  # Take first middle file
                logger.info(f"Using middle JSON file: {json_file}")
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    elements = self._convert_mineru_middle_json_to_elements(data)
                    
                    # Enhance with model.json data if available
                    if model_json_files:
                        logger.info(f"Enhancing with model JSON file: {model_json_files[0]}")
                        async with aiofiles.open(model_json_files[0], 'r', encoding='utf-8') as model_f:
                            model_content = await model_f.read()
                            try:
                                model_data = json.loads(model_content)
                                elements = self._enhance_elements_with_model_data(elements, model_data)
                            except json.JSONDecodeError:
                                logger.warning(f"Could not parse model output file as JSON: {model_json_files[0]}. Skipping enhancement.")
                                logger.debug(f"Model file content (first 500 chars): {model_content[:500]}")
                    
                    sections, extracted_title, extracted_abstract = self._extract_structured_content_from_elements(elements)
            # Parse content_list.json output if available (fallback)
            elif content_list_files:
                json_file = content_list_files[0]  # Take first content_list file
                logger.info(f"Using content_list file: {json_file}")
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    elements = self._convert_mineru_content_list_to_elements(data)
                    sections, extracted_title, extracted_abstract = self._extract_structured_content_from_elements(elements)
            # Fallback to other JSON files
            elif json_files:
                json_file = json_files[0]  # Take first JSON file
                logger.info(f"Using JSON file: {json_file}")
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    elements = self._convert_mineru_json_to_elements(data)
                    sections, extracted_title, extracted_abstract = self._extract_structured_content_from_elements(elements)
            
            # Parse Markdown output if available and try to extract title/abstract from it
            if md_files:
                md_file = md_files[0]  # Take first markdown file
                async with aiofiles.open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = await f.read()
                    # Try to extract title and abstract from markdown if not already found
                    if not extracted_title or not extracted_abstract:
                        md_title, md_abstract = self._extract_title_abstract_from_markdown(markdown_content)
                        extracted_title = extracted_title or md_title
                        extracted_abstract = extracted_abstract or md_abstract
            
            # If no specific output found, create basic structure
            if not elements and not markdown_content:
                # Create minimal response
                elements = [MinerUElement(
                    type="text",
                    content="Processing completed but no structured output available",
                    bbox=BoundingBox(x=0, y=0, width=0, height=0),
                    pageNumber=1,
                    confidence=0.5
                )]
                markdown_content = "# Document processed\n\nProcessing completed successfully."
            
            # Create structured content with extracted information
            structured_content = StructuredContent(
                title=extracted_title,  # Use extracted title instead of hardcoded
                abstract=extracted_abstract,  # Use extracted abstract
                sections=sections
            )
            
            # Create metadata
            metadata = ProcessingMetadata(
                totalPages=len(set(elem.pageNumber for elem in elements)) if elements else 1,
                processingTimeMs=0,  # Will be set by caller
                detectedLanguage=options.lang or "auto",
                documentType="research_paper"
            )
            
            return ProcessingResponse(
                success=True,
                elements=elements,
                markdown=markdown_content,
                structuredContent=structured_content,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error parsing MinerU output: {e}")
            # Return basic successful response even if parsing fails
            return ProcessingResponse(
                success=True,
                elements=[],
                markdown="# Processing completed\n\nDocument was processed but detailed extraction failed.",
                structuredContent=StructuredContent(title="Processed Document", sections=[]),
                metadata=ProcessingMetadata(
                    totalPages=1,
                    processingTimeMs=0,
                    detectedLanguage="auto",
                    documentType="unknown"
                )
            )
    
    def _convert_mineru_middle_json_to_elements(self, data) -> List[MinerUElement]:
        """Convert MinerU middle JSON output to our element format"""
        elements = []
        
        logger.debug(f"Received middle JSON data of type: {type(data)}")
        if isinstance(data, dict):
            logger.debug(f"Middle JSON data keys: {list(data.keys())}")
        elif isinstance(data, list):
            logger.debug(f"Middle JSON data is a list of length: {len(data)}")

        pdf_info = []
        if isinstance(data, dict) and "pdf_info" in data:
            pdf_info = data["pdf_info"]
        elif isinstance(data, list):
            pdf_info = data  # Handle cases where middle.json is a list of pages
        
        if not pdf_info:
            logger.warning("Could not find processable content in middle JSON. It's neither a dict with 'pdf_info' nor a list of pages.")
            return elements

        logger.debug(f"Processing {len(pdf_info)} pages from pdf_info")

        for page_idx, page_data in enumerate(pdf_info):
            page_blocks = []
            if isinstance(page_data, dict):
                if "para_blocks" in page_data:
                    page_blocks = page_data["para_blocks"]
                    logger.debug(f"Page {page_idx}: found {len(page_blocks)} para_blocks")
                elif "preproc_blocks" in page_data:
                    page_blocks = page_data["preproc_blocks"]
                    logger.debug(f"Page {page_idx}: found {len(page_blocks)} preproc_blocks")
            
            if not page_blocks:
                logger.debug(f"Page {page_idx}: no 'para_blocks' or 'preproc_blocks' found.")
                continue

            for block in page_blocks:
                if not isinstance(block, dict):
                    continue
                        
                element_type = block.get("type", "text")
                bbox_array = block.get("bbox", [0, 0, 0, 0])
                block_index = block.get("index", 0)
                
                # Convert bbox from [x1, y1, x2, y2] to our format
                if len(bbox_array) >= 4:
                    x1, y1, x2, y2 = bbox_array[:4]
                    # Convert to normalized coordinates (assuming page dimensions)
                    # For now, we'll use absolute coordinates and normalize later
                    bbox = BoundingBox(
                        x=float(x1), 
                        y=float(y1), 
                        width=float(x2 - x1), 
                        height=float(y2 - y1)
                    )
                else:
                    bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)
                
                # Handle image blocks specially
                if element_type == "image":
                    # Look for image_body and image_caption in blocks
                    image_content = ""
                    image_caption = ""
                    
                    if "blocks" in block:
                        for sub_block in block["blocks"]:
                            if sub_block.get("type") == "image_body":
                                # Extract image path
                                for line in sub_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        if span.get("type") == "image" and "image_path" in span:
                                            image_content = span["image_path"]
                                
                            elif sub_block.get("type") == "image_caption":
                                # Extract caption text
                                for line in sub_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        if "content" in span:
                                            image_caption += span["content"] + " "
                    
                    # Create separate elements for image and caption
                    if image_content:
                        image_element = MinerUElement(
                            type="image",
                            content=image_content,
                            bbox=bbox,
                            pageNumber=page_idx + 1,
                            hierarchy=None,
                            confidence=0.95,
                            metadata=None
                        )
                        image_element.__dict__["_mineru_metadata"] = {
                            "source": "mineru_middle_json",
                            "original_type": "image_body",
                            "block_index": block_index
                        }
                        elements.append(image_element)
                    
                    if image_caption.strip():
                        caption_element = MinerUElement(
                            type="caption",
                            content=image_caption.strip(),
                            bbox=bbox,  # Same bbox for now, could be refined
                            pageNumber=page_idx + 1,
                            hierarchy=None,
                            confidence=0.95,
                            metadata=ElementMetadata(figureCaption=image_caption.strip())
                        )
                        caption_element.__dict__["_mineru_metadata"] = {
                            "source": "mineru_middle_json",
                            "original_type": "image_caption",
                            "block_index": block_index
                        }
                        elements.append(caption_element)
                
                elif element_type == "table":
                    table_html = ""
                    table_caption = ""
                    table_footnote = ""
                    
                    if "blocks" in block:
                        for sub_block in block["blocks"]:
                            sub_block_type = sub_block.get("type")
                            
                            # Extract HTML from table_body
                            if sub_block_type == "table_body":
                                for line in sub_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        if span.get("type") == "table" and "html" in span:
                                            table_html += span["html"]
                            
                            # Extract caption text
                            elif sub_block_type == "table_caption":
                                for line in sub_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        if "content" in span:
                                            table_caption += span["content"] + " "

                            # Extract footnote text
                            elif sub_block_type == "table_footnote":
                                for line in sub_block.get("lines", []):
                                    for span in line.get("spans", []):
                                        if "content" in span:
                                            table_footnote += span["content"] + " "

                    if table_html:
                        table_element = MinerUElement(
                            type="table",
                            content=table_html.strip(),
                            bbox=bbox,
                            pageNumber=page_idx + 1,
                            hierarchy=None,
                            confidence=0.95,
                            metadata=ElementMetadata(
                                tableCaption=table_caption.strip() if table_caption else None
                            )
                        )
                        table_element.__dict__["_mineru_metadata"] = {
                            "source": "mineru_middle_json",
                            "original_type": "table",
                            "block_index": block_index,
                            "footnote": table_footnote.strip() if table_footnote else None
                        }
                        elements.append(table_element)
                
                else:
                    # Handle text, title, and other elements
                    text_content = ""
                    current_metadata = {}

                    if "lines" in block:
                        for line in block["lines"]:
                            if isinstance(line, dict) and "spans" in line:
                                for span in line["spans"]:
                                    if isinstance(span, dict) and "content" in span:
                                        span_type = span.get("type", "text")
                                        span_content = span.get("content", "").strip()
                                        
                                        # Handle equations specifically
                                        if span_type in ["equation", "inline_equation"] and span_content:
                                            # If there's pending text, save it first
                                            if text_content:
                                                element = MinerUElement(
                                                    type="text",
                                                    content=text_content.strip(),
                                                    bbox=bbox, # Note: bbox is for the whole block
                                                    pageNumber=page_idx + 1,
                                                    hierarchy=None,
                                                    confidence=0.95,
                                                    metadata=ElementMetadata(**current_metadata)
                                                )
                                                elements.append(element)
                                                text_content = ""
                                                current_metadata = {}

                                            # Create and add the equation element
                                            equation_element = MinerUElement(
                                                type="equation",
                                                content=span_content,
                                                bbox=bbox, # Note: needs refinement for span-level bbox
                                                pageNumber=page_idx + 1,
                                                hierarchy=None,
                                                confidence=0.95,
                                                metadata=ElementMetadata(equationType=span_type)
                                            )
                                            equation_element.__dict__["_mineru_metadata"] = {
                                                "source": "mineru_middle_json",
                                                "original_type": span_type,
                                                "block_index": block_index
                                            }
                                            elements.append(equation_element)
                                        
                                        # Append regular text
                                        else:
                                            text_content += span["content"] + " "


                    if text_content: # Process any remaining text in the block
                        text_content = text_content.strip()
                        if text_content:
                            # Determine if this is likely a heading based on type and content
                            if element_type == "title" or self._looks_like_heading(text_content):
                                element_type = "heading"
                            
                            element = MinerUElement(
                                type=element_type,
                                content=text_content,
                                bbox=bbox,
                                pageNumber=page_idx + 1,
                                hierarchy=None,  # Will be determined later based on content
                                confidence=0.95,
                                metadata=None  # We'll store metadata separately for now
                            )
                            # Store metadata as a custom attribute for our processing
                            element.__dict__["_mineru_metadata"] = {
                                "source": "mineru_middle_json",
                                "original_type": block.get("type", "unknown"),
                                "block_index": block_index
                            }
                            elements.append(element)
        
        # Sort elements by page and block index for proper order
        elements.sort(key=lambda x: (x.pageNumber, getattr(x, '_mineru_metadata', {}).get("block_index", 0)))
        
        logger.info(f"Converted {len(elements)} elements from MinerU middle JSON")
        return elements
    
    def _enhance_elements_with_model_data(self, elements: List[MinerUElement], model_data: dict) -> List[MinerUElement]:
        """Enhance elements with raw model detection data for better bounding boxes and classification"""
        
        if not isinstance(model_data, list) or not model_data:
            logger.warning("Model data is not in expected format")
            return elements
            
        # Get layout detections from all pages
        all_layout_dets = []
        for page_data in model_data:
            if isinstance(page_data, dict):
                layout_dets = page_data.get("layout_dets", [])
                page_info = page_data.get("page_info", {})
                page_no = page_info.get("page_no", 0)
                
                # Add page number to each detection
                for det in layout_dets:
                    det_with_page = dict(det)
                    det_with_page["page_no"] = page_no
                    all_layout_dets.append(det_with_page)
                    
        logger.info(f"Processing {len(all_layout_dets)} layout detections from model across all pages")
        
        # Official MinerU category mapping from documentation
        category_mapping = {
            0: "title",           # Title
            1: "text",            # Plain text
            2: "abandon",         # Headers, footers, page numbers, page annotations
            3: "image",           # Figure/Image
            4: "caption",         # Figure caption
            5: "table",           # Table
            6: "caption",         # Table caption
            7: "caption",         # Table footnote
            8: "equation",        # Block formula (isolate_formula)
            9: "caption",         # Formula caption
            13: "equation",       # Inline formula (embedding)
            14: "equation",       # Block formula (isolated)
            15: "text"            # OCR recognition result
        }
        
        enhanced_elements = []
        
        # First, add model-detected elements that might not be in middle.json
        for detection in all_layout_dets:
            if detection.get("score", 0) < 0.7:  # Skip low-confidence detections
                continue
                
            category_id = detection.get("category_id")
            poly = detection.get("poly", [])
            page_no = detection.get("page_no", 0)
            
            # Skip "abandon" category (headers, footers, page numbers)
            if category_id == 2:
                continue
            
            if len(poly) >= 8:
                # Convert polygon to bounding box
                # poly format: [x0, y0, x1, y1, x2, y2, x3, y3] (top-left, top-right, bottom-right, bottom-left)
                x_coords = [poly[i] for i in range(0, len(poly), 2)]
                y_coords = [poly[i] for i in range(1, len(poly), 2)]
                
                x1, x2 = min(x_coords), max(x_coords)
                y1, y2 = min(y_coords), max(y_coords)
                
                bbox = BoundingBox(
                    x=float(x1),
                    y=float(y1), 
                    width=float(x2 - x1),
                    height=float(y2 - y1)
                )
                
                element_type = category_mapping.get(category_id, "text")
                confidence = detection.get("score", 0.9)
                
                # Check if this detection overlaps significantly with existing elements
                overlaps_existing = False
                for existing_element in elements:
                    if (existing_element.pageNumber == page_no + 1 and 
                        self._bbox_overlap_ratio(bbox, existing_element.bbox) > 0.7):
                        overlaps_existing = True
                        break
                
                # Add special elements that might be missed (tables, equations, figures)
                if not overlaps_existing and element_type in ["image", "table", "equation"]:
                    content = detection.get("latex", detection.get("html", f"{element_type.title()} detected by model"))
                    
                    model_element = MinerUElement(
                        type=element_type,
                        content=content,
                        bbox=bbox,
                        pageNumber=page_no + 1,  # Convert 0-based to 1-based
                        hierarchy=None,
                        confidence=confidence,
                        metadata=None
                    )
                    model_element.__dict__["_mineru_metadata"] = {
                        "source": "mineru_model_json",
                        "category_id": category_id,
                        "model_score": confidence,
                        "detection_method": "cv_model",
                        "has_latex": bool(detection.get("latex")),
                        "has_html": bool(detection.get("html"))
                    }
                    enhanced_elements.append(model_element)
        
        # Enhance existing elements with model data for better bounding boxes
        for element in elements:
            enhanced_element = element
            best_match = None
            best_overlap = 0
            
            # Find the best matching model detection on the same page
            for detection in all_layout_dets:
                if detection.get("score", 0) < 0.5:
                    continue
                
                page_no = detection.get("page_no", 0)
                if page_no + 1 != element.pageNumber:  # Must be on same page
                    continue
                    
                poly = detection.get("poly", [])
                if len(poly) >= 8:
                    x_coords = [poly[i] for i in range(0, len(poly), 2)]
                    y_coords = [poly[i] for i in range(1, len(poly), 2)]
                    
                    x1, x2 = min(x_coords), max(x_coords)
                    y1, y2 = min(y_coords), max(y_coords)
                    
                    model_bbox = BoundingBox(
                        x=float(x1), y=float(y1),
                        width=float(x2 - x1), height=float(y2 - y1)
                    )
                    
                    overlap = self._bbox_overlap_ratio(element.bbox, model_bbox)
                    if overlap > best_overlap and overlap > 0.3:
                        best_overlap = overlap
                        best_match = {
                            "bbox": model_bbox,
                            "score": detection.get("score", 0),
                            "category_id": detection.get("category_id"),
                            "latex": detection.get("latex"),
                            "html": detection.get("html")
                        }
            
            # Enhance element with model data if good match found
            if best_match and best_overlap > 0.5:
                # Use LaTeX or HTML content if available for equations
                enhanced_content = element.content
                enhanced_metadata = ElementMetadata()
                
                if best_match.get("latex") and element.type in ["equation", "formula"]:
                    enhanced_content = best_match["latex"]
                elif best_match.get("html") and element.type in ["table"]:
                    enhanced_metadata = ElementMetadata()  # Could add HTML to metadata later
                
                enhanced_element = MinerUElement(
                    type=element.type,
                    content=enhanced_content,
                    bbox=best_match["bbox"],  # Use more precise model bbox
                    pageNumber=element.pageNumber,
                    hierarchy=element.hierarchy,
                    confidence=max(element.confidence, best_match["score"]),
                    metadata=enhanced_metadata
                )
                
                # Copy original metadata and add enhancements
                original_meta = getattr(element, '_mineru_metadata', {})
                enhanced_element.__dict__["_mineru_metadata"] = {
                    **original_meta,
                    "model_enhanced": True,
                    "model_score": best_match["score"],
                    "bbox_overlap": best_overlap,
                    "model_category_id": best_match["category_id"]
                }
                if best_match.get("html"):
                    enhanced_element.__dict__["_mineru_metadata"]["html_content"] = best_match["html"]
            
            enhanced_elements.append(enhanced_element)
        
        logger.info(f"Enhanced {len(elements)} elements with model data, total: {len(enhanced_elements)}")
        return enhanced_elements
    
    def _bbox_overlap_ratio(self, bbox1: BoundingBox, bbox2: BoundingBox) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        # Calculate intersection
        x1_inter = max(bbox1.x, bbox2.x)
        y1_inter = max(bbox1.y, bbox2.y)
        x2_inter = min(bbox1.x + bbox1.width, bbox2.x + bbox2.width)
        y2_inter = min(bbox1.y + bbox1.height, bbox2.y + bbox2.height)
        
        if x2_inter <= x1_inter or y2_inter <= y1_inter:
            return 0.0
        
        intersection_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
        
        # Calculate union
        area1 = bbox1.width * bbox1.height
        area2 = bbox2.width * bbox2.height
        union_area = area1 + area2 - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area
    
    def _convert_mineru_json_to_elements(self, data: Dict) -> List[MinerUElement]:
        """Convert MinerU JSON output to our element format"""
        elements = []
        
        # This is a placeholder - actual implementation would depend on 
        # MinerU's specific JSON output format which we'll discover after successful run
        if isinstance(data, dict):
            # Handle different possible MinerU output structures
            for key, value in data.items():
                if isinstance(value, (list, dict)):
                    # Process nested structures
                    continue
                    
        return elements
    
    def _extract_structured_content_from_elements(self, elements: List[MinerUElement]) -> tuple[List[Section], Optional[str], Optional[str]]:
        """Extract structured content (sections, title, abstract) from MinerU elements"""
        sections = []
        current_section = None
        extracted_title = None
        extracted_abstract = None
        
        # First pass: look for title and abstract patterns
        for i, element in enumerate(elements):
            content_lower = element.content.lower().strip()
            
            # Look for title (prioritize elements marked as "title" type)
            if not extracted_title and element.type in ["title", "heading"]:
                mineru_meta = getattr(element, '_mineru_metadata', {})
                # For middle.json elements, trust the "title" type more
                if (mineru_meta.get("source") == "mineru_middle_json" and
                    mineru_meta.get("original_type") == "title"):
                    extracted_title = element.content.strip()
                # For model.json elements, trust category_id 0 (title)
                elif (mineru_meta.get("source") == "mineru_model_json" and
                      mineru_meta.get("category_id") == 0):
                    extracted_title = element.content.strip()
                elif element.type == "heading" and len(element.content.strip()) < 200:
                    extracted_title = element.content.strip()
            
            # Fallback title detection for other sources
            elif not extracted_title and element.type in ["text"]:
                # Skip very short content, page numbers, headers/footers
                if (len(element.content.strip()) > 10 and 
                    not content_lower.isdigit() and 
                    not content_lower.startswith(('page ', 'doi:', 'arxiv:')) and
                    element.pageNumber <= 2):  # Title usually on first couple pages
                    
                    # Higher priority for elements with title-like characteristics
                    if (any(keyword in content_lower for keyword in ['title', 'abstract']) or
                        (i < 5 and len(element.content.strip()) < 200)):  # Early elements, reasonable length
                        extracted_title = element.content.strip()
            
            # Look for abstract
            if not extracted_abstract and element.type == "text":
                if (content_lower.startswith('abstract') or 
                    'abstract' in content_lower[:50] or
                    (extracted_title and i > 0 and i < 10 and len(element.content.strip()) > 100)):
                    
                    # Clean up abstract text
                    abstract_text = element.content.strip()
                    if abstract_text.lower().startswith('abstract'):
                        # Remove "abstract" prefix and clean up
                        abstract_text = abstract_text[8:].strip()
                        if abstract_text.startswith(':') or abstract_text.startswith('-'):
                            abstract_text = abstract_text[1:].strip()
                    
                    if len(abstract_text) > 50:  # Reasonable abstract length
                        extracted_abstract = abstract_text
        
        # Second pass: build sections from headings
        for element in elements:
            if element.type in ["heading", "title"] or self._looks_like_heading(element.content):
                # Start new section
                if current_section:
                    sections.append(current_section)
                
                # Determine heading level
                level = 1
                if element.hierarchy and element.hierarchy.level:
                    level = element.hierarchy.level
                else:
                    # Try to infer level from content patterns
                    content = element.content.strip()
                    if content.count('.') >= 2:  # Like "1.2.3 Subsection"
                        level = content.count('.') + 1
                    elif content[0:1].isdigit():  # Like "1. Section"
                        level = 2
                
                current_section = Section(
                    title=element.content.strip(),
                    level=level,
                    content="",
                    elements=[element]
                )
            elif current_section and element.type == "text":
                current_section.elements.append(element)
                current_section.content += element.content + "\n"
        
        # Add final section
        if current_section:
            sections.append(current_section)
            
        return sections, extracted_title, extracted_abstract
    
    def _looks_like_heading(self, text: str) -> bool:
        """Determine if text looks like a section heading"""
        text = text.strip()
        if len(text) > 200:  # Too long to be a heading
            return False
        
        # Common heading patterns
        heading_patterns = [
            r'^\d+\.?\s+[A-Z]',  # "1. Introduction" or "1 Introduction"
            r'^\d+\.\d+\.?\s+[A-Z]',  # "1.1 Background" 
            r'^[A-Z][A-Z\s]{5,50}$',  # "INTRODUCTION", "RELATED WORK"
            r'^[A-Z][a-z]+(\s[A-Z][a-z]*)*$',  # "Introduction", "Related Work"
        ]
        
        import re
        return any(re.match(pattern, text) for pattern in heading_patterns)
    
    def _extract_title_abstract_from_markdown(self, markdown_content: str) -> tuple[Optional[str], Optional[str]]:
        """Extract title and abstract from markdown content"""
        lines = markdown_content.split('\n')
        extracted_title = None
        extracted_abstract = None
        
        # Look for title (first # heading or first substantial line)
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Title from markdown heading
            if line.startswith('# ') and not extracted_title:
                extracted_title = line[2:].strip()
                continue
            
            # Title from first substantial line
            if not extracted_title and len(line) > 10 and not line.startswith(('#', '*', '-', '>')):
                extracted_title = line
                continue
            
            # Abstract detection
            if 'abstract' in line.lower() and not extracted_abstract:
                # Abstract might be on same line or following lines
                if ':' in line:
                    abstract_text = line.split(':', 1)[1].strip()
                    if len(abstract_text) > 20:
                        extracted_abstract = abstract_text
                else:
                    # Look at next few lines for abstract content
                    abstract_lines = []
                    for j in range(i + 1, min(i + 10, len(lines))):
                        next_line = lines[j].strip()
                        if not next_line or next_line.startswith('#'):
                            break
                        abstract_lines.append(next_line)
                    
                    if abstract_lines:
                        extracted_abstract = ' '.join(abstract_lines)
                
                if extracted_abstract and len(extracted_abstract) > 300:
                    # Truncate very long abstracts
                    extracted_abstract = extracted_abstract[:300] + "..."
        
        return extracted_title, extracted_abstract

    def _build_sections_from_elements(self, elements: List[MinerUElement]) -> List[Section]:
        """Build sections from elements (legacy method, kept for compatibility)"""
        sections, _, _ = self._extract_structured_content_from_elements(elements)
        return sections

    def _convert_mineru_content_list_to_elements(self, data) -> List[MinerUElement]:
        """Convert MinerU content_list JSON output to our element format (fallback)"""
        elements = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if not isinstance(item, dict):
                    continue

                element_type = item.get('type', 'text')
                page_idx = item.get('page_idx', item.get('page', item.get('pageNumber', 0)))
                
                # Bbox is not always present in content_list, so handle it gracefully
                bbox_info = item.get('bbox', {})
                if isinstance(bbox_info, dict):
                    bbox = BoundingBox(
                        x=bbox_info.get('x', 0.0),
                        y=bbox_info.get('y', 0.0),
                        width=bbox_info.get('width', 0.0),
                        height=bbox_info.get('height', 0.0)
                    )
                else:
                    bbox = BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0)

                # Handle images and their captions
                if element_type == 'image':
                    img_path = item.get('img_path')
                    if img_path:
                        image_element = MinerUElement(
                            type="image",
                            content=img_path,
                            bbox=bbox,
                            pageNumber=page_idx + 1,
                            hierarchy=None,
                            confidence=item.get('confidence', 0.9),
                            metadata=None
                        )
                        image_element.__dict__["_mineru_metadata"] = {
                            'source': 'mineru_content_list',
                            'original_idx': idx,
                            'original_type': 'image'
                        }
                        elements.append(image_element)

                    img_captions = item.get('img_caption', [])
                    if img_captions and isinstance(img_captions, list):
                        caption_text = ' '.join(img_captions).strip()
                        if caption_text:
                            caption_element = MinerUElement(
                                type="caption",
                                content=caption_text,
                                bbox=bbox,  # Use same bbox for now
                                pageNumber=page_idx + 1,
                                hierarchy=None,
                                confidence=item.get('confidence', 0.9),
                                metadata=ElementMetadata(figureCaption=caption_text)
                            )
                            caption_element.__dict__["_mineru_metadata"] = {
                                'source': 'mineru_content_list',
                                'original_idx': idx,
                                'original_type': 'image_caption'
                            }
                            elements.append(caption_element)
                    continue  # Move to the next item

                # Handle text-based elements
                text_content = (
                    item.get('text', '') or 
                    item.get('content', '') or 
                    item.get('markdown', '') or
                    str(item.get('value', ''))
                ).strip()
                
                if not text_content:
                    continue
                
                # Use text_level for more reliable heading detection
                if item.get('text_level') == 1 and element_type in ('text', 'span'):
                    element_type = 'heading'
                elif element_type in ('text', 'span'):
                    if self._looks_like_heading(text_content):
                        element_type = 'heading'
                    elif any(keyword in text_content.lower() for keyword in ['figure', 'fig.', 'table', 'tab.']):
                        element_type = 'caption'
                
                element = MinerUElement(
                    type=element_type,
                    content=text_content,
                    bbox=bbox,
                    pageNumber=page_idx + 1,  # Convert 0-based to 1-based page numbering
                    hierarchy=None,
                    confidence=item.get('confidence', 0.9),  # High confidence for MinerU results
                    metadata=None
                )
                # Store metadata as a custom attribute for our processing
                element.__dict__["_mineru_metadata"] = {
                    'source': 'mineru_content_list',
                    'original_idx': idx,
                    'original_type': item.get('type', 'unknown')
                }
                elements.append(element)
        
        logger.info(f"Converted {len(elements)} elements from MinerU content_list")
        return elements


    async def process_pdf_raw(self, pdf_path: str, options: ProcessingOptions, force: bool = False) -> Dict:
        """Process PDF and return raw MinerU output files"""
        
        if not MINERU_AVAILABLE:
            raise HTTPException(status_code=503, detail="MinerU not available")
        
        # Create temporary output directory
        output_dir = self.temp_dir / f"raw_output_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        output_dir.mkdir(exist_ok=True)
        
        try:
            # Build and execute MinerU command
            cmd = self._build_mineru_command(pdf_path, str(output_dir), options)
            logger.info(f"Executing raw command: {' '.join(cmd)}")
            
            # Run MinerU command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(output_dir)
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                raise ValueError(f"MinerU processing failed: {error_msg}")
            
            # Collect all output files
            files = {}
            file_count = 0
            
            for file_path in output_dir.rglob("*"):
                if file_path.is_file():
                    file_count += 1
                    relative_path = str(file_path.relative_to(output_dir))
                    
                    try:
                        # Try to read as text first
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            files[relative_path] = {
                                "type": "text",
                                "content": content,
                                "size": len(content)
                            }
                    except UnicodeDecodeError:
                        # If not text, read as binary and base64 encode
                        with open(file_path, 'rb') as f:
                            content = f.read()
                            files[relative_path] = {
                                "type": "binary",
                                "content": base64.b64encode(content).decode(),
                                "size": len(content)
                            }
                    except Exception as e:
                        files[relative_path] = {
                            "type": "error",
                            "content": str(e),
                            "size": 0
                        }
            
            return {
                "success": True,
                "file_count": file_count,
                "files": files,
                "stdout": stdout.decode() if stdout else "",
                "stderr": stderr.decode() if stderr else "",
            }
            
        finally:
            # Cleanup
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)

# Processor instance
processor = MinerUProcessor()

# Utility functions
def get_cache_key(file_content: bytes) -> str:
    import hashlib
    return hashlib.sha256(file_content).hexdigest()

async def get_cached_result(cache_key: str) -> Optional[ProcessingResponse]:
    if not redis_client:
        return None
    try:
        cached = await asyncio.to_thread(redis_client.get, f"mineru:result:{cache_key}")
        if cached:
            return ProcessingResponse.parse_raw(cached)
    except Exception as e:
        logger.warning(f"Cache retrieval error: {e}")
    return None

async def cache_result(cache_key: str, result: ProcessingResponse):
    if not redis_client:
        return
    try:
        await asyncio.to_thread(
            redis_client.setex,
            f"mineru:result:{cache_key}",
            settings.cache_ttl,
            result.json()
        )
    except Exception as e:
        logger.warning(f"Cache storage error: {e}")

# API Routes
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    sglang_status = await check_sglang_server(settings.sglang_url)
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mineru_available": MINERU_AVAILABLE,
        "redis_available": redis_client is not None,
        "sglang_server_available": sglang_status,
        "sglang_url": settings.sglang_url
    }

@app.get("/status")
async def get_status():
    """Get service status and capabilities"""
    sglang_status = await check_sglang_server(settings.sglang_url)
    return {
        "service": "MinerU PDF Processing Service",
        "version": "1.0.0",
        "mineru_version": MINERU_VERSION,
        "mineru_available": MINERU_AVAILABLE,
        "redis_available": redis_client is not None,
        "sglang_server_available": sglang_status,
        "sglang_url": settings.sglang_url,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "cache_ttl_seconds": settings.cache_ttl,
        "supported_backends": ["pipeline", "vlm-transformers", "vlm-sglang-engine", "vlm-sglang-client"],
        "supported_methods": ["auto", "txt", "ocr"],
        "supported_formats": ["pdf"],
        "default_backend": settings.mineru_backend
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(
    background_tasks: BackgroundTasks,
    request: Request,  # Move this before default arguments
    pdf: UploadFile = File(...),
    options: str = '{}',
    force: str = '0',
    _auth: bool = Depends(verify_api_key)
):
    """Process PDF with MinerU using SGLang acceleration"""
    
    # Validate file
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    # Check file size
    content = await pdf.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(
            status_code=413, 
            detail=f"File too large. Maximum size: {settings.max_file_size // (1024 * 1024)}MB"
        )
    
    # Check force in query params too
    query_force = request.query_params.get('force', 'false').lower() in ('true', '1', 'yes')
    
    # Parse options
    try:
        options_dict = json.loads(options) if options else {}
        
        # 🚀 Set default backend to SGLang client if not specified
        if 'backend' not in options_dict:
            options_dict['backend'] = 'vlm-sglang-client'
        
        # Add detailed debugging
        logger.info(f"[DEBUG] Raw options string: '{options}'")
        logger.info(f"[DEBUG] Parsed options_dict: {options_dict}")
        logger.info(f"[DEBUG] Backend selected: {options_dict.get('backend', 'default')}")
        
        processing_options = ProcessingOptions(**options_dict)
        # Check all three sources for force parameter
        force_reprocess = (
            options_dict.get('force', False) or 
            force.lower() in ('1', 'true', 'yes') or 
            query_force
        )
        
        logger.info(f"🚀 Processing with backend: {processing_options.backend}")
        if processing_options.backend == 'vlm-sglang-client':
            logger.info(f"🚀 SGLang server URL: {processing_options.sglang_url}")
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options: {e}")
    
    # Check cache (skip if force reprocess)
    cache_key = get_cache_key(content + options.encode())
    cached_result = None if force_reprocess else await get_cached_result(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return cached_result
    
    # Save uploaded file temporarily
    temp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            f.write(content)
            temp_pdf = f.name
        
        # Process with MinerU
        result = await processor.process_pdf(temp_pdf, processing_options)
        
        # Cache result
        background_tasks.add_task(cache_result, cache_key, result)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        # Enhanced error logging
        if hasattr(e, 'last_attempt') and hasattr(e.last_attempt, 'exception'):
            underlying_error = e.last_attempt.exception()
            logger.error(f"Processing error (after {e.last_attempt.attempt_number} retries): {underlying_error}")
            if isinstance(underlying_error, HTTPException):
                raise underlying_error
            else:
                raise HTTPException(status_code=500, detail=f"Processing failed: {str(underlying_error)}")
        else:
            logger.error(f"Processing error: {e}")
            raise HTTPException(status_code=500, detail=f"Internal processing error: {str(e)}")
    finally:
        # Cleanup
        if temp_pdf and os.path.exists(temp_pdf):
            os.unlink(temp_pdf)

@app.post("/process-pdf-raw")
async def process_pdf_raw(
    pdf: UploadFile = File(...),
    options: str = '{}',
    force: str = '0',
    _auth: bool = Depends(verify_api_key)
):
    """Process PDF and return raw MinerU output files"""
    
    # Validate file
    if not pdf.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    
    content = await pdf.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {settings.max_file_size // (1024 * 1024)}MB")
    
    # Parse options
    try:
        options_dict = json.loads(options) if options else {}
        # Default to SGLang client
        if 'backend' not in options_dict:
            options_dict['backend'] = 'vlm-sglang-client'
        processing_options = ProcessingOptions(**options_dict)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options: {e}")
    
    # Check force parameter
    force_reprocess = force.lower() in ('1', 'true', 'yes')
    
    # Save uploaded file temporarily
    temp_pdf = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as f:
            f.write(content)
            temp_pdf = f.name
        
        # Process and get raw files
        result = await processor.process_pdf_raw(temp_pdf, processing_options, force_reprocess)
        return result
        
    except Exception as e:
        logger.error(f"Raw processing error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_pdf and os.path.exists(temp_pdf):
            os.unlink(temp_pdf)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)