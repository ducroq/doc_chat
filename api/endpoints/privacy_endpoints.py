from logging import getLogger
import pathlib

from fastapi import APIRouter
from fastapi.responses import HTMLResponse

router = APIRouter()
logger = getLogger(__name__)

@router.get("/privacy", response_class=HTMLResponse)
async def privacy_notice():
    """
    Serve the privacy notice.
    
    Returns:
        HTMLResponse: HTML content of the privacy notice
    """
    try:
        privacy_path = pathlib.Path("privacy_notice.html")
        if privacy_path.exists():
            return privacy_path.read_text(encoding="utf-8")
        else:
            logger.warning("privacy_notice.html not found, serving fallback notice")
            return """
            <!DOCTYPE html>
            <html>
                <head>
                    <title>Privacy Notice</title>
                    <style>
                        body { font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto; padding: 20px; }
                    </style>
                </head>
                <body>
                    <h1>Chat Logging Privacy Notice</h1>
                    <p>When enabled, this system logs interactions for research purposes.</p>
                    <p>All data is processed in accordance with GDPR. Logs are automatically deleted after 30 days.</p>
                    <p>Please contact the system administrator for more information or to request deletion of your data.</p>
                </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving privacy notice: {str(e)}")
        return "<h1>Privacy Notice</h1><p>Error loading privacy notice.</p>"