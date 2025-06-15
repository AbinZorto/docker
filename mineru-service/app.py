import os
import asyncio
import tempfile
import shutil
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

# MinerU imports (using official API)
try:
    import magic_pdf
    from magic_pdf.pipe.UNIPipe import UNIPipe
    from magic_pdf.pipe.OCRPipe import OCRPipe
    from magic_pdf.pipe.TXTpipe import TXTpipe
    from magic_pdf.model.model_list import AtomicModel
    MINERU_AVAILABLE = True
    MINERU_VERSION = magic_pdf.__version__
    logger.info(f"✅ MinerU {MINERU_VERSION} loaded successfully")
except ImportError as e:
    logger.warning(f"❌ MinerU not available: {e}")
    MINERU_AVAILABLE = False
    MINERU_VERSION = "Not installed"

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

# MinerU Processor
class MinerUProcessor:
    def __init__(self):
        self.temp_dir = Path("/tmp/mineru_processing")
        self.temp_dir.mkdir(exist_ok=True)
        
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    async def process_pdf(self, pdf_path: str, options: ProcessingOptions) -> ProcessingResponse:
        """Process PDF using MinerU"""
        start_time = datetime.now()
        
        if not MINERU_AVAILABLE:
            raise HTTPException(status_code=503, detail="MinerU not available")
        
        try:
            # Initialize MinerU components
            parser = PdfParser()
            layout_analyzer = LayoutAnalyzer() if options.layoutAnalysis else None
            structure_analyzer = StructureAnalyzer()
            
            logger.info(f"Processing PDF: {pdf_path}")
            
            # Parse PDF
            document = await asyncio.to_thread(parser.parse, pdf_path)
            if not document:
                raise ValueError("Failed to parse PDF document")
            
            # Layout analysis
            if layout_analyzer and options.layoutAnalysis:
                document = await asyncio.to_thread(layout_analyzer.analyze, document)
            
            # Structure analysis for academic papers
            structured_doc = await asyncio.to_thread(structure_analyzer.analyze, document)
            
            # Extract elements
            elements = []
            sections = []
            markdown_content = ""
            
            for page_num, page in enumerate(document.pages, 1):
                page_elements = self._extract_page_elements(page, page_num, options)
                elements.extend(page_elements)
            
            # Build structured content
            if structured_doc:
                sections = self._build_sections(structured_doc)
                markdown_content = self._generate_markdown(structured_doc)
            
            # Generate response
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return ProcessingResponse(
                success=True,
                elements=elements,
                markdown=markdown_content,
                structuredContent=StructuredContent(
                    title=getattr(structured_doc, 'title', None),
                    abstract=getattr(structured_doc, 'abstract', None),
                    sections=sections
                ),
                metadata=ProcessingMetadata(
                    totalPages=len(document.pages),
                    processingTimeMs=int(processing_time),
                    detectedLanguage=getattr(document, 'language', 'en'),
                    documentType=self._detect_document_type(structured_doc)
                )
            )
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            return ProcessingResponse(
                success=False,
                elements=[],
                markdown="",
                structuredContent=StructuredContent(sections=[]),
                metadata=ProcessingMetadata(
                    totalPages=0,
                    processingTimeMs=0,
                    detectedLanguage="unknown",
                    documentType="other"
                ),
                error=str(e)
            )
    
    def _extract_page_elements(self, page, page_num: int, options: ProcessingOptions) -> List[MinerUElement]:
        """Extract elements from a single page"""
        elements = []
        
        # Text elements
        for text_block in getattr(page, 'text_blocks', []):
            element = MinerUElement(
                type="text",
                content=text_block.text,
                bbox=BoundingBox(
                    x=text_block.bbox[0] / page.width,
                    y=text_block.bbox[1] / page.height,
                    width=(text_block.bbox[2] - text_block.bbox[0]) / page.width,
                    height=(text_block.bbox[3] - text_block.bbox[1]) / page.height
                ),
                pageNumber=page_num,
                confidence=getattr(text_block, 'confidence', 0.9),
                metadata=ElementMetadata(
                    fontSize=getattr(text_block, 'font_size', None),
                    fontWeight=getattr(text_block, 'font_weight', None),
                    isItalic=getattr(text_block, 'is_italic', None)
                )
            )
            elements.append(element)
        
        # Image elements
        if options.extractImages:
            for image in getattr(page, 'images', []):
                # Convert image to base64 if needed
                image_data = None
                if hasattr(image, 'data'):
                    image_data = base64.b64encode(image.data).decode('utf-8')
                
                element = MinerUElement(
                    type="image",
                    content=f"data:image/png;base64,{image_data}" if image_data else "",
                    bbox=BoundingBox(
                        x=image.bbox[0] / page.width,
                        y=image.bbox[1] / page.height,
                        width=(image.bbox[2] - image.bbox[0]) / page.width,
                        height=(image.bbox[3] - image.bbox[1]) / page.height
                    ),
                    pageNumber=page_num,
                    confidence=getattr(image, 'confidence', 0.8),
                    metadata=ElementMetadata(
                        imageData=image_data,
                        figureCaption=getattr(image, 'caption', None)
                    )
                )
                elements.append(element)
        
        # Table elements
        for table in getattr(page, 'tables', []):
            element = MinerUElement(
                type="table",
                content=getattr(table, 'text', str(table)),
                bbox=BoundingBox(
                    x=table.bbox[0] / page.width,
                    y=table.bbox[1] / page.height,
                    width=(table.bbox[2] - table.bbox[0]) / page.width,
                    height=(table.bbox[3] - table.bbox[1]) / page.height
                ),
                pageNumber=page_num,
                confidence=getattr(table, 'confidence', 0.85),
                metadata=ElementMetadata(
                    tableCaption=getattr(table, 'caption', None)
                )
            )
            elements.append(element)
        
        return elements
    
    def _build_sections(self, structured_doc) -> List[Section]:
        """Build sections from structured document"""
        sections = []
        for section in getattr(structured_doc, 'sections', []):
            sections.append(Section(
                title=section.title,
                level=getattr(section, 'level', 1),
                content=section.content,
                elements=[]  # Would need to map elements to sections
            ))
        return sections
    
    def _generate_markdown(self, structured_doc) -> str:
        """Generate markdown from structured document"""
        markdown = ""
        
        if hasattr(structured_doc, 'title'):
            markdown += f"# {structured_doc.title}\n\n"
        
        if hasattr(structured_doc, 'abstract'):
            markdown += f"## Abstract\n\n{structured_doc.abstract}\n\n"
        
        for section in getattr(structured_doc, 'sections', []):
            level = "#" * (getattr(section, 'level', 1) + 1)
            markdown += f"{level} {section.title}\n\n{section.content}\n\n"
        
        return markdown
    
    def _detect_document_type(self, structured_doc) -> str:
        """Detect document type based on structure"""
        if hasattr(structured_doc, 'abstract'):
            return "research_paper"
        return "article"

# Global processor instance
processor = MinerUProcessor()

# Cache utilities
def get_cache_key(file_content: bytes) -> str:
    """Generate cache key from file content"""
    import hashlib
    return f"mineru_{hashlib.md5(file_content).hexdigest()}"

async def get_cached_result(cache_key: str) -> Optional[ProcessingResponse]:
    """Get cached processing result"""
    if not redis_client:
        return None
    
    try:
        cached = await asyncio.to_thread(redis_client.get, cache_key)
        if cached:
            return ProcessingResponse.parse_raw(cached)
    except Exception as e:
        logger.warning(f"Cache get error: {e}")
    
    return None

async def cache_result(cache_key: str, result: ProcessingResponse):
    """Cache processing result"""
    if not redis_client:
        return
    
    try:
        await asyncio.to_thread(
            redis_client.setex,
            cache_key,
            settings.cache_ttl,
            result.json()
        )
    except Exception as e:
        logger.warning(f"Cache set error: {e}")

# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "mineru_available": MINERU_AVAILABLE,
        "redis_available": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/status")
async def get_status():
    """Get service status"""
    return {
        "version": "1.0.0",
        "status": "running",
        "uptime": "unknown",  # Would need to track actual uptime
        "mineru_available": MINERU_AVAILABLE,
        "features": {
            "ocr": True,
            "layout_analysis": True,
            "image_extraction": True,
            "caching": redis_client is not None
        }
    }

@app.post("/process-pdf", response_model=ProcessingResponse)
async def process_pdf(
    background_tasks: BackgroundTasks,
    pdf: UploadFile = File(...),
    options: str = '{}',
    _auth: bool = Depends(verify_api_key)
):
    """Process PDF file using MinerU"""
    
    # Validate file
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="File must be a PDF")
    
    # Read file content
    content = await pdf.read()
    if len(content) > settings.max_file_size:
        raise HTTPException(status_code=413, detail="File too large")
    
    # Parse options
    try:
        processing_options = ProcessingOptions.parse_raw(options)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid options: {e}")
    
    # Check cache
    cache_key = get_cache_key(content)
    cached_result = await get_cached_result(cache_key)
    if cached_result:
        logger.info("Returning cached result")
        return cached_result
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(content)
        temp_path = temp_file.name
    
    try:
        # Process PDF
        result = await processor.process_pdf(temp_path, processing_options)
        
        # Cache result in background
        if result.success:
            background_tasks.add_task(cache_result, cache_key, result)
        
        return result
        
    finally:
        # Clean up
        os.unlink(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug
    ) 