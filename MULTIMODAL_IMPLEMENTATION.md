# Multimodal Input Implementation

## Overview

This document describes the comprehensive file, image, and audio input handling implementation for the OpenRouter Responses API pipe.

## Features Implemented

### ✅ Image Input Support
- **Base64 images**: Automatically saved to OWUI storage
- **Remote image URLs**: Downloaded and saved locally to prevent data loss
- **Supported formats** (per OpenRouter docs):
  - `image/png`
  - `image/jpeg`
  - `image/webp`
  - `image/gif`
- **Detail levels**: Preserves `auto`, `low`, `high` detail settings
- **Size limit**: Configurable via `REMOTE_FILE_MAX_SIZE_MB` valve (default: 50MB)
- **Base64 validation**: Configurable via `BASE64_MAX_SIZE_MB` valve (default: 50MB)

### ✅ File Input Support
- **Multiple input formats**: Handles nested and flat block structures
- **All Responses API fields** (per OpenAPI spec):
  - `file_id`: Reference to uploaded file
  - `file_data`: Base64 or data URL content
  - `filename`: Filename for model context
  - `file_url`: URL to file
- **Base64 files**: Automatically saved to OWUI storage
- **Remote file URLs**: Downloaded and saved locally
- **Size limit**: Configurable via `REMOTE_FILE_MAX_SIZE_MB` valve (default: 50MB)
- **Base64 validation**: Configurable via `BASE64_MAX_SIZE_MB` valve (default: 50MB)

### ✅ Audio Input Support
- **Format conversion**: Transforms multiple input formats to Responses API format
- **Supported audio formats** (per OpenRouter docs):
  - `wav`
  - `mp3`
- **MIME type mapping**:
  - `audio/mpeg` → `mp3`
  - `audio/mp3` → `mp3`
  - `audio/wav` → `wav`
  - `audio/wave` → `wav`
  - `audio/x-wav` → `wav`
- **Base64 required**: Audio must be base64-encoded (URLs NOT supported per OpenRouter docs)

### ✅ Retry Logic for Remote Downloads
- **Exponential backoff**: Automatic retry with increasing delays for transient failures
- **Configurable via valves**:
  - `REMOTE_DOWNLOAD_MAX_RETRIES`: Maximum retry attempts (default: 3, range: 0-10)
  - `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS`: Initial delay before first retry (default: 5s, range: 1-60s)
  - `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS`: Maximum total retry time (default: 45s, range: 5-300s)
  - `REMOTE_FILE_MAX_SIZE_MB`: Maximum file size for downloads (default: 50MB, range: 1-500MB). When RAG uploads are enabled in Open WebUI, the effective limit auto-aligns with the configured `FILE_MAX_SIZE`.
  - `BASE64_MAX_SIZE_MB`: Maximum base64 data size (default: 50MB, range: 1-500MB)
- **Smart retry logic**:
  - Retries network errors, HTTP 5xx errors, and 429 rate limits
  - Does not retry HTTP 4xx client errors (except 429)
  - Logs each retry attempt with timing information
  - Respects remote servers with exponential backoff delays

### ✅ Production-Ready Size Limits
- **Configurable limits**: No magic numbers, all limits via valves
- **File download size**: `REMOTE_FILE_MAX_SIZE_MB` (default: 50MB, up to 500MB) with automatic alignment to Open WebUI's `FILE_MAX_SIZE` when RAG uploads are enabled
- **Base64 data size**: `BASE64_MAX_SIZE_MB` (default: 50MB, up to 500MB)
- **Early validation**: Base64 size checked before decoding to prevent memory issues
- **Clear error messages**: Size limit violations logged with specific valve values
- **Flexible deployment**: Adjust limits per deployment scenario (dev/staging/prod)

## Architecture

### Helper Methods

#### `_get_user_by_id(user_id: str)`
Fetches user record from database for file upload operations.
- Uses `run_in_threadpool` to avoid blocking async operations
- Returns `None` on failure (logged but doesn't crash)

#### `_get_file_by_id(file_id: str)`
Looks up file metadata from Open WebUI's file storage.
- Uses `run_in_threadpool` to avoid blocking async operations
- Returns `None` on failure (logged but doesn't crash)

#### `_upload_to_owui_storage(request, user, file_data, filename, mime_type)`
Uploads file or image to Open WebUI storage and returns internal URL.
- **Purpose**: Prevents data loss from:
  - Remote URLs that may become inaccessible
  - Base64 data that is temporary
  - External image hosts that delete images after a period
- **Processing**: Disabled (`process=False`) to avoid unnecessary overhead
- Returns internal URL path (e.g., `/api/v1/files/{id}`)
- Returns `None` on failure (logged but doesn't crash)

#### `_download_remote_url(url, timeout_seconds=None)`
Downloads file or image from remote HTTP/HTTPS URL with automatic retry using exponential backoff.

**Retry Configuration (via valves)**:
- `REMOTE_DOWNLOAD_MAX_RETRIES`: Maximum retry attempts (default: 3, range: 0-10)
- `REMOTE_DOWNLOAD_INITIAL_RETRY_DELAY_SECONDS`: Initial delay before first retry (default: 5s, range: 1-60s)
- `REMOTE_DOWNLOAD_MAX_RETRY_TIME_SECONDS`: Maximum total retry time (default: 45s, range: 5-300s)
- `REMOTE_FILE_MAX_SIZE_MB`: Maximum file size (default: 50MB, range: 1-500MB)
- Uses exponential backoff: `delay * 2^attempt`

**Retryable Errors**:
- Network errors (connection timeout, DNS failure, etc.)
- HTTP 5xx server errors
- HTTP 429 rate limit errors

**Non-Retryable Errors**:
- HTTP 4xx client errors (except 429)
- Invalid URLs
- Files exceeding configured size limit

**Additional Features**:
- **Size limit**: Configurable via `REMOTE_FILE_MAX_SIZE_MB` valve (default: 50MB); automatically capped to Open WebUI's `FILE_MAX_SIZE` when RAG uploads are enabled
- **Timeout**: Capped at 60 seconds per attempt to prevent hanging requests
- **MIME type normalization**: `image/jpg` → `image/jpeg`
- **Retry logging**: Logs each retry attempt with timing information
- Returns dict with `data`, `mime_type`, `url` or `None` on failure

#### `_parse_data_url(data_url: str)`
Extracts base64 data from data URL with size validation.
- **Format**: `data:<mime_type>;base64,<base64_data>`
- **MIME type normalization**: `image/jpg` → `image/jpeg`
- **Size validation**: Validates size before decoding via `BASE64_MAX_SIZE_MB` valve
- **Early rejection**: Prevents memory issues by validating before decoding
- Returns dict with `data`, `mime_type`, `b64` or `None` on failure

#### `_validate_base64_size(b64_data: str)`
Validates base64 data size is within configured limits.
- **Purpose**: Prevent memory issues from huge base64 payloads
- **Validation**: Estimates decoded size before actual decoding
- **Limit**: Configured via `BASE64_MAX_SIZE_MB` valve (default: 50MB)
- **Early warning**: Logs size violations with specific limit values
- Returns `True` if within limits, `False` if too large

#### `_emit_status(event_emitter, message, done=False)`
Emits status updates to the Open WebUI client.
- **Status indicators**:
  - 📥 Download/upload in progress
  - ✅ Successful completion
  - ⚠️ Warning or non-critical error
  - 🔴 Critical error
- No-op if `event_emitter` is `None`

### Transformer Functions

#### `_to_input_image(block: dict) -> dict`
Converts Open WebUI image blocks into Responses API format.

**Processing flow**:
1. Extract image URL from block (nested or flat structure)
2. If data URL: Parse, upload to OWUI storage
3. If remote URL: Download, upload to OWUI storage
4. If OWUI file reference: Keep as-is
5. Return Responses API format with detail level

**Error handling**: All errors caught and logged with status emissions. Returns empty `image_url` rather than crashing.

#### `_to_input_file(block: dict) -> dict`
Converts Open WebUI file blocks into Responses API format.

**Processing flow**:
1. Extract fields from nested or flat block structure
2. If `file_data` is data URL: Parse, upload to OWUI storage, set `file_url`
3. If `file_data` is remote URL: Download, upload to OWUI storage, set `file_url`
4. If `file_id`: Keep as-is (already in OWUI storage)
5. Return all available fields to Responses API

**Error handling**: All errors caught and logged with status emissions. Returns minimal valid block rather than crashing.

#### `_to_input_audio(block: dict) -> dict`
Converts Open WebUI audio blocks into Responses API format.

**Input formats handled**:
1. Chat Completions: `{"type": "input_audio", "input_audio": "<base64>"}`
2. Tool output: `{"type": "audio", "mimeType": "audio/mp3", "data": "<base64>"}`
3. Already correct: `{"type": "input_audio", "input_audio": {"data": "...", "format": "..."}}`

**Processing flow**:
1. Extract audio payload from block
2. If already in Responses API format: Return as-is
3. If string payload: Determine format from MIME type, wrap in Responses format
4. If invalid: Return empty audio block with default format

**Error handling**: All errors caught and logged with status emissions. Returns minimal valid block rather than crashing.

## Block Type Mapping

| Open WebUI Type | Responses API Type | Transformer | Notes |
|----------------|-------------------|-------------|-------|
| `text` | `input_text` | Lambda | Simple passthrough |
| `image_url` | `input_image` | `_to_input_image` | Download & save |
| `input_file` | `input_file` | `_to_input_file` | Full field support |
| `file` | `input_file` | `_to_input_file` | Chat Completions format |
| `input_audio` | `input_audio` | `_to_input_audio` | Format conversion |
| `audio` | `input_audio` | `_to_input_audio` | Tool output format |

## Error Handling Philosophy

**Nothing can crash the pipe.** All operations follow this pattern:

```python
try:
    # Attempt operation
    result = await dangerous_operation()
except Exception as exc:
    self.logger.error(f"Operation failed: {exc}")
    await self._emit_error(
        event_emitter,
        f"⚠️ Operation failed: {exc}",
        show_error_message=False
    )
    # Return fallback/minimal valid data
    return fallback_value
```

This ensures:
- Errors are logged for debugging
- Users see helpful status messages
- Processing continues for other blocks
- No data loss from crashes

## Size Limits

### Configurable Size Limits (Production-Ready)

**Philosophy**: All size limits are configurable via valves to support different deployment scenarios (development, staging, production) without code changes.

**No Magic Numbers**: All limits defined through Pydantic Field validators with sensible defaults and safe bounds.

### Remote File/Image Downloads

**Valve**: `REMOTE_FILE_MAX_SIZE_MB`
- **Default**: 50MB (more generous than original 10MB hardcoded limit)
- **Range**: 1-500MB (configurable per deployment needs)
- **Purpose**: Prevents excessive network usage and storage consumption
- **Applied to**: Both remote image URLs and remote file URLs

**Enforcement**:
```python
max_size_bytes = self.valves.REMOTE_FILE_MAX_SIZE_MB * 1024 * 1024
if len(response.content) > max_size_bytes:
    # Log warning with specific limit
    # Return None to skip file
```

### Base64 Data Validation

**Valve**: `BASE64_MAX_SIZE_MB`
- **Default**: 50MB
- **Range**: 1-500MB (configurable per deployment needs)
- **Purpose**: Prevent memory issues from huge base64 payloads before decoding
- **Applied to**: All base64 data URLs (images, files, audio)

**Early Validation**:
```python
# Validate BEFORE decoding to prevent memory issues
if not self._validate_base64_size(b64_data):
    return None  # Size check failed
```

**Why Separate Limits**:
- Remote downloads: Network and storage impact
- Base64 validation: Memory and HTTP request size impact
- Different deployment scenarios may need different limits

**Files/Images Exceeding Limits**:
- Rejected with descriptive warning log including configured limit
- Return `None` from helper methods
- Processing continues for other content blocks without crashing

## Documentation Compliance

### OpenRouter Image Documentation
**Source**: `C:\Work\Dev\openrouter_docs\manual\docs\guides\overview\multimodal\images.md`

- ✅ Supports both URLs and base64 data URLs
- ✅ Supported formats: png, jpeg, webp, gif
- ✅ `image/jpg` normalized to `image/jpeg`

### OpenRouter Audio Documentation
**Source**: `C:\Work\Dev\openrouter_docs\manual\docs\guides\overview\multimodal\audio.md`

- ✅ Audio must be base64-encoded (URLs NOT supported)
- ✅ Supported formats: wav, mp3
- ✅ Format specified in `input_audio.format` field

### OpenRouter Responses API Specification
**Source**: `C:\Work\Dev\openrouter_docs\manual\docs\api\api-reference\responses\create-responses.md`

- ✅ `input_image`: type, image_url, detail (optional)
- ✅ `input_file`: type, file_id (optional), file_data (optional), filename (optional), file_url (optional)
- ✅ `input_audio`: type, input_audio.data, input_audio.format

## Testing

### Test Coverage

Created comprehensive test suite in `tests/test_multimodal_inputs.py`:

- **Helper method tests**:
  - Data URL parsing
  - Remote URL downloading
  - Size limit enforcement
  - MIME type normalization

- **Transformer tests**:
  - Image URL and base64 handling
  - File input field preservation
  - Audio format conversion
  - Error handling and fallbacks

- **Integration tests**:
  - Combined multimodal inputs
  - Multiple images/files in single message
  - Error isolation (one block failure doesn't crash others)

- **Compliance tests**:
  - OpenRouter format support verification
  - Documentation requirement validation

### Running Tests

```bash
cd c:\Work\Dev\openrouter_responses_pipe
python -m pytest tests/test_multimodal_inputs.py -v
```

## Usage Examples

### Image with Base64 Data

**Input**:
```json
{
  "role": "user",
  "content": [
    {"type": "text", "text": "What's in this image?"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KG..."}}
  ]
}
```

**Processing**:
1. Parse base64 data URL
2. Upload to OWUI storage
3. Emit status: "📥 Saved base64 image to storage"

**Output**:
```json
{
  "type": "message",
  "role": "user",
  "content": [
    {"type": "input_text", "text": "What's in this image?"},
    {"type": "input_image", "image_url": "/api/v1/files/abc123", "detail": "auto"}
  ]
}
```

### File with Remote URL

**Input**:
```json
{
  "role": "user",
  "content": [
    {"type": "input_file", "file_data": "https://example.com/document.pdf", "filename": "document.pdf"}
  ]
}
```

**Processing**:
1. Download from remote URL
2. Upload to OWUI storage
3. Emit status: "📥 Downloaded and saved file: document.pdf"

**Output**:
```json
{
  "type": "message",
  "role": "user",
  "content": [
    {"type": "input_file", "file_url": "/api/v1/files/def456", "filename": "document.pdf"}
  ]
}
```

### Audio from Tool Output

**Input**:
```json
{
  "role": "user",
  "content": [
    {"type": "audio", "mimeType": "audio/wav", "data": "UklGRiQAAABXQVZFZm10..."}
  ]
}
```

**Processing**:
1. Extract base64 data
2. Map MIME type to format: `audio/wav` → `wav`
3. Wrap in Responses API format

**Output**:
```json
{
  "type": "message",
  "role": "user",
  "content": [
    {"type": "input_audio", "input_audio": {"data": "UklGRiQAAABXQVZFZm10...", "format": "wav"}}
  ]
}
```

## Changes Made

### Files Modified

1. **`openrouter_responses_pipe.py`**:
   - Added imports: `httpx`, `io`, `uuid`, `UploadFile`, `BackgroundTasks`, `Headers`, `Files`, `Users`, `upload_file_handler`, `tenacity`
   - Added 5 configurable valves for retry logic and size limits (no magic numbers)
   - Added helper methods for multimodal input processing
   - Enhanced image transformer with download/upload support
   - Added file transformer with full field support
   - Added audio transformer with format conversion
   - Removed audio blocker to enable audio support
   - Updated requirements: added `httpx`, `tenacity`

### Files Created

2. **`tests/test_multimodal_inputs.py`**:
   - Comprehensive test suite for all multimodal functionality
   - 50+ test cases covering helpers, transformers, integration, and compliance

3. **`MULTIMODAL_IMPLEMENTATION.md`** (this file):
   - Complete implementation documentation
   - Architecture overview
   - Usage examples
   - Documentation compliance verification

4. **`PRODUCTION_READINESS_AUDIT.md`**:
   - Production readiness analysis
   - Identification of issues and solutions
   - Design philosophy documentation (chat history persistence)
   - Recommendations for deployment

## Benefits

### For Users
- ✅ **No data loss**: Images and files are permanently stored
- ✅ **Remote URL support**: Can use any publicly accessible image/file URL
- ✅ **Audio support**: Can send audio to 10 audio-capable models
- ✅ **Error resilience**: Pipe never crashes from multimodal input errors

### For Models
- ✅ **Complete context**: Models receive all available file metadata (filename, etc.)
- ✅ **Persistent access**: Images/files remain accessible even if original sources disappear
- ✅ **Format compliance**: All inputs correctly formatted per Responses API spec

### For Developers
- ✅ **Comprehensive docs**: Detailed docstrings on all methods
- ✅ **Test coverage**: Extensive test suite for reliability
- ✅ **Error logging**: Clear error messages for debugging
- ✅ **Status updates**: Real-time feedback via status emitters
- ✅ **Configurable limits**: All size limits and retry behavior configurable via valves
- ✅ **No magic numbers**: Production-ready with sensible defaults and safe bounds

## Future Enhancements (Not Implemented)

- **Video input**: Not supported by Responses API (only Chat Completions)
- **Image output**: Not supported by Responses API (only 1 model supports it anyway)
- **Caching downloads**: Could cache downloaded files to avoid re-downloading identical URLs
- **PDF plugin support**: Could expose OpenRouter's PDF processing engines via valve (pdf-text, mistral-ocr, native)

## Conclusion

This implementation provides complete, production-ready multimodal input support for the OpenRouter Responses API pipe, with:
- Full compliance with OpenRouter documentation
- Robust error handling with exponential backoff retry logic ensuring the pipe never crashes
- Persistent storage preventing data loss (chat history works forever)
- Comprehensive testing and documentation
- Configurable size limits and retry behavior (no magic numbers)
- Support for 130+ image models, 43+ file models, and 10+ audio models

All features are grounded in official OpenRouter documentation and OpenAPI specifications, with deployment flexibility through configurable valves.
