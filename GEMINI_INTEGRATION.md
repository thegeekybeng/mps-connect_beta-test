# Gemini AI Integration for MPS Connect

This document describes the integration of Google's Gemini AI models into the MPS Connect system, providing enhanced AI capabilities across all three core components.

## Overview

The Gemini integration uses a two-tier approach:

- **Gemini Flash**: Fast, cost-effective model for quick previews and drafts
- **Gemini Pro**: High-quality model for final, polished outputs

## Components Enhanced

### 1. Case Analysis & Classification

- **Flash**: Quick case analysis and initial categorization
- **Pro**: Detailed analysis with confidence scores and recommendations

### 2. Letter Generation

- **Flash**: Quick letter drafts and templates
- **Pro**: Polished, professional letters with accurate details

### 3. Approval Workflow & Recommendations

- **Flash**: Quick approval recommendations
- **Pro**: Detailed reasoning and risk assessment

## Setup

### 1. Environment Variables

Add to your environment configuration:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

### 2. Dependencies

The integration automatically installs the required dependency:

```bash
pip install google-generativeai==0.3.2
```

### 3. API Endpoints

New endpoints are available at:

- `/api/gemini/analyze-case-preview` - Quick case analysis
- `/api/gemini/analyze-case-final` - Detailed case analysis
- `/api/gemini/generate-letter-preview` - Quick letter draft
- `/api/gemini/generate-letter-final` - Polished letter
- `/api/gemini/recommend-approval-preview` - Quick approval recommendation
- `/api/gemini/recommend-approval-final` - Detailed approval recommendation

## Usage

### Backend Integration

```python
from api.gemini_integration import get_gemini_integration

# Get the Gemini integration instance
gemini = get_gemini_integration()

# Analyze case with Flash (preview)
analysis = await gemini.analyze_case_preview("Case text here")

# Analyze case with Pro (final)
analysis = await gemini.analyze_case_final("Case text here", "Additional feedback")

# Generate letter with Flash (preview)
letter = await gemini.generate_letter_preview(case_data)

# Generate letter with Pro (final)
letter = await gemini.generate_letter_final(case_data, "User feedback")
```

### Frontend Integration

```javascript
// Initialize Gemini integration
const gemini = new GeminiIntegration(
  "https://your-api-url.com",
  "your-api-key"
);
const geminiUI = new GeminiUI(gemini);
geminiUI.init();

// Use the enhanced UI with Gemini buttons
// The UI will automatically add buttons for:
// - Quick Analysis (Flash)
// - Detailed Analysis (Pro)
// - Quick Letter Draft (Flash)
// - Final Letter (Pro)
```

## API Reference

### Case Analysis

#### Preview (Flash)

```http
POST /api/gemini/analyze-case-preview
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_text": "Case description here"
}
```

#### Final (Pro)

```http
POST /api/gemini/analyze-case-final
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_text": "Case description here",
    "feedback": "Additional context or feedback"
}
```

### Letter Generation

#### Preview (Flash)

```http
POST /api/gemini/generate-letter-preview
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_data": {
        "case_text": "Case description",
        "constituent_name": "John Doe",
        "issue_type": "Traffic Fine"
    }
}
```

#### Final (Pro)

```http
POST /api/gemini/generate-letter-final
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_data": {
        "case_text": "Case description",
        "constituent_name": "John Doe",
        "issue_type": "Traffic Fine"
    },
    "feedback": "Make it more formal and include specific details"
}
```

### Approval Recommendations

#### Preview (Flash)

```http
POST /api/gemini/recommend-approval-preview
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_analysis": {
        "categories": [...],
        "priority_level": "HIGH",
        "key_facts": [...]
    }
}
```

#### Final (Pro)

```http
POST /api/gemini/recommend-approval-final
Content-Type: application/json
X-API-Key: your-api-key

{
    "case_analysis": {
        "categories": [...],
        "priority_level": "HIGH",
        "key_facts": [...]
    },
    "feedback": "Consider additional risk factors"
}
```

## Response Format

All endpoints return a consistent response format:

```json
{
  "success": true,
  "data": {
    // Specific data based on endpoint
  },
  "error": null,
  "model_used": "flash" // or "pro"
}
```

## Error Handling

The integration includes comprehensive error handling:

- Graceful fallbacks when Gemini is unavailable
- Detailed error messages for debugging
- Automatic retry logic for transient failures

## Cost Optimization

- **Flash Model**: Used for previews and quick iterations
- **Pro Model**: Used only for final outputs
- **Caching**: Results are cached to avoid duplicate API calls
- **Rate Limiting**: Built-in rate limiting to control costs

## Security

- API key is stored securely in environment variables
- All requests require authentication via X-API-Key header
- Input validation and sanitization
- Safe error handling to prevent information leakage

## Monitoring

The integration includes logging for:

- API call success/failure rates
- Response times
- Error patterns
- Cost tracking

## Troubleshooting

### Common Issues

1. **"Gemini AI not available"**

   - Check that GEMINI_API_KEY is set
   - Verify the API key is valid
   - Check network connectivity

2. **"Gemini integration not initialized"**

   - Ensure google-generativeai is installed
   - Check for import errors in logs

3. **Poor quality results**
   - Try providing more detailed feedback
   - Use Pro model for final outputs
   - Check input data quality

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('api.gemini_integration').setLevel(logging.DEBUG)
```

## Future Enhancements

- [ ] Custom model fine-tuning
- [ ] Multi-language support
- [ ] Advanced prompt engineering
- [ ] Integration with other AI models
- [ ] Real-time collaboration features
- [ ] Advanced analytics and insights

## Support

For issues or questions about the Gemini integration:

1. Check the logs for error details
2. Verify API key and configuration
3. Test with simple inputs first
4. Contact the development team with specific error messages

