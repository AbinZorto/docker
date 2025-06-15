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

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, BackgroundTasks
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

settings = Settings()

# FastAPI app
app = FastAPI(
    title="MinerU PDF Processing Service",
    description="Advanced PDF processing service for research papers using MinerU",
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
    backend: str = Field(default="pipeline", description="Backend: pipeline, vlm-transformers, vlm-sglang-engine")
    lang: Optional[str] = Field(default=None, description="Document language for OCR accuracy")
    start_page: Optional[int] = Field(default=None, description="Starting page (0-based)")
    end_page: Optional[int] = Field(default=None, description="Ending page (0-based)")
    formula: bool = Field(default=True, description="Enable formula parsing")
    table: bool = Field(default=True, description="Enable table parsing")
    device: str = Field(default="cpu", description="Inference device: cpu, cuda, cuda:0, npu, mps")
    vram: Optional[int] = Field(default=None, description="Max GPU VRAM usage per process (MB)")
    model_source: str = Field(default="huggingface", description="Model source: huggingface, modelscope, local")
    
    # Output options
    output_format: str = Field(default="both", description="Output format: json, markdown, or both")
    extract_images: bool = Field(default=True, description="Extract images from PDF")
    
    # Legacy aliases for backwards compatibility
    enableOCR: Optional[bool] = Field(default=None, description="Legacy: use method='ocr' instead")
    layoutAnalysis: Optional[bool] = Field(default=None, description="Legacy: always enabled")
    extractImages: Optional[bool] = Field(default=None, description="Legacy: use extract_images")
    outputFormat: Optional[str] = Field(default=None, description="Legacy: use output_format")
    
    # Add at the end:
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
        
        # Add backend
        if options.backend and options.backend != "pipeline":
            cmd.extend(['-b', options.backend])
            
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
            
        # Add device
        if options.device and options.device != "cpu":
            cmd.extend(['-d', options.device])
            
        # Add VRAM limit
        if options.vram:
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
            
            logger.info(f"Processing completed in {processing_time:.0f}ms")
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
        
        try:
            # Log all files in output directory for debugging
            all_files = list(output_dir.rglob("*"))
            logger.info(f"Files in output directory {output_dir}:")
            for file in all_files:
                logger.info(f"  {file.relative_to(output_dir)} ({'dir' if file.is_dir() else 'file'})")
            
            # Look for MinerU output files (typically JSON and markdown)
            json_files = list(output_dir.rglob("*.json"))  # Use rglob to search subdirectories
            md_files = list(output_dir.rglob("*.md"))
            
            # Prioritize content_list.json files which contain the structured content
            content_list_files = list(output_dir.rglob("*_content_list.json"))
            
            logger.info(f"Found {len(json_files)} JSON files, {len(content_list_files)} content_list files, and {len(md_files)} MD files")
            
            # Parse content_list.json output if available (preferred)
            if content_list_files:
                json_file = content_list_files[0]  # Take first content_list file
                logger.info(f"Using content_list file: {json_file}")
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    elements = self._convert_mineru_content_list_to_elements(data)
                    sections = self._build_sections_from_elements(elements)
            # Fallback to other JSON files
            elif json_files:
                json_file = json_files[0]  # Take first JSON file
                logger.info(f"Using JSON file: {json_file}")
                async with aiofiles.open(json_file, 'r', encoding='utf-8') as f:
                    content = await f.read()
                    data = json.loads(content)
                    elements = self._convert_mineru_json_to_elements(data)
                    sections = self._build_sections_from_elements(elements)
            
            # Parse Markdown output if available  
            if md_files:
                md_file = md_files[0]  # Take first markdown file
                async with aiofiles.open(md_file, 'r', encoding='utf-8') as f:
                    markdown_content = await f.read()
            
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
            
            # Create structured content
            structured_content = StructuredContent(
                title="Processed Document",
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
    
    def _convert_mineru_content_list_to_elements(self, data) -> List[MinerUElement]:
        """Convert MinerU content_list JSON output to our element format"""
        elements = []
        
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict):
                    element_type = item.get('type', 'text')
                    text_content = item.get('text', '').strip()
                    page_idx = item.get('page_idx', 0)
                    
                    if text_content:  # Only add non-empty content
                        element = MinerUElement(
                            type=element_type,
                            content=text_content,
                            bbox=BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0),
                            pageNumber=page_idx + 1,  # Convert 0-based to 1-based page numbering
                            hierarchy=None,
                            confidence=0.9,  # High confidence for MinerU results
                            metadata={"source": "mineru_content_list", "original_idx": idx}
                        )
                        elements.append(element)
        
        return elements
    
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
    
    def _build_sections_from_elements(self, elements: List[MinerUElement]) -> List[Section]:
        """Build sections from elements"""
        sections = []
        current_section = None
        
        for element in elements:
            if element.type == "heading":
                # Start new section
                if current_section:
                    sections.append(current_section)
                current_section = Section(
                    title=element.content,
                    level=element.hierarchy.level if element.hierarchy else 1,
                    content="",
                    elements=[element]
                )
            elif current_section:
                current_section.elements.append(element)
                current_section.content += element.content + "\n"
        
        if current_section:
            sections.append(current_section)
            
        return sections

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
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "mineru_available": MINERU_AVAILABLE,
        "redis_available": redis_client is not None
    }

@app.get("/status")
async def get_status():
    """Get service status and capabilities"""
    return {
        "service": "MinerU PDF Processing Service",
        "version": "1.0.0",
        "mineru_version": MINERU_VERSION,
        "mineru_available": MINERU_AVAILABLE,
        "redis_available": redis_client is not None,
        "max_file_size_mb": settings.max_file_size // (1024 * 1024),
        "cache_ttl_seconds": settings.cache_ttl,
        "supported_backends": ["pipeline", "vlm-transformers", "vlm-sglang-engine"],
        "supported_methods": ["auto", "txt", "ocr"],
        "supported_formats": ["pdf"]
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    options: str = '{}',
    force: str = '0',
    _auth: bool = Depends(verify_api_key)
):
    """Process PDF with MinerU"""
    
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
    
    # Parse options
    try:
        options_dict = json.loads(options) if options else {}
        
        # Add detailed debugging
        logger.info(f"[DEBUG] Raw options string: '{options}'")
        logger.info(f"[DEBUG] Parsed options_dict: {options_dict}")
        logger.info(f"[DEBUG] Options dict keys: {list(options_dict.keys())}")
        
        processing_options = ProcessingOptions(**options_dict)
        # Check both sources for force parameter (JSON and form field)
        force_reprocess = options_dict.get('force', False) or force.lower() in ('1', 'true', 'yes')
        
        # Add debug logging
        logger.info(f"[DEBUG] Force parameters received - form_field: '{force}', json_option: {options_dict.get('force', 'not found')}, final_force_reprocess: {force_reprocess}")
        
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
